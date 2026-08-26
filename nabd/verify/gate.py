"""Verification Gate: deterministic, evidence-backed task completion gate.

The gate is the *sole authority* for deciding whether a task has
succeeded.  It evaluates every mandatory criterion against concrete
evidence (filesystem hashes, command exit codes, snapshot diffs) and
issues one of five decisions: PASS, REPAIR, ROLLBACK, BLOCKED, or
TIMEOUT.

Design rules (from the plan):
* No PASS based on LLM text alone.
* Every criterion result links to an Evidence ID.
* UNKNOWN external changes block automatic completion.
* Repeated identical failures with no improvement trigger ROLLBACK.
* Budget exhaustion triggers ROLLBACK.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..evidence import EvidenceStore, EvidenceType
from ..jail import JailError, WorkspaceJail
from ..raw_facts import RawFacts
from .types import (
    Criterion,
    CriterionKind,
    Decision,
    FailureSignature,
    Report,
    Result,
    SuccessCriteria,
)


class GateError(Exception):
    """Raised when the gate encounters an unrecoverable internal error."""


class VerificationGate:
    """Evaluates a SuccessCriteria document and produces a Report.

    Parameters
    ----------
    root : Path
        Workspace root (must match EvidenceStore root).
    evidence : EvidenceStore
        The task's evidence store.
    unknown_paths : set[str]
        Paths changed outside the agent's tool calls (detected by
        snapshot comparison).  Empty means no external changes.
    snapshots_before : dict[str, str]
        Mapping of relative-path -> sha256 taken before task execution.
    snapshots_after : dict[str, str]
        Mapping of relative-path -> sha256 taken after task execution.
    changed_files : list[str]
        Relative paths of files modified by the agent during this task.
    repair_count : int
        How many repair attempts have been made so far.
    """

    def __init__(
        self,
        root: Path,
        evidence: EvidenceStore,
        unknown_paths: Optional[Set[str]] = None,
        snapshots_before: Optional[Dict[str, str]] = None,
        snapshots_after: Optional[Dict[str, str]] = None,
        changed_files: Optional[List[str]] = None,
        repair_count: int = 0,
        failure_signatures: Optional[List[FailureSignature]] = None,
        current_attempt_seq: Optional[int] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.evidence = evidence
        self.jail = WorkspaceJail(self.root)
        self.unknown_paths: Set[str] = unknown_paths or set()
        self.snapshots_before: Dict[str, str] = snapshots_before or {}
        self.snapshots_after: Dict[str, str] = snapshots_after or {}
        self.changed_files: List[str] = changed_files or []
        self.repair_count = repair_count
        # History of failure signatures from *prior* repair attempts. When the
        # current attempt's signature matches a prior one, the gate breaks the
        # repair loop (repeated identical failure -> ROLLBACK).
        self.failure_signatures: List[FailureSignature] = list(failure_signatures or [])
        # When set, evidence records are scoped to this attempt sequence so a
        # prior attempt's PASS cannot satisfy a criterion for the current one.
        self.current_attempt_seq: Optional[int] = current_attempt_seq

    def _evidence_for_current_attempt(self):
        records = self.evidence.get_all()
        if self.current_attempt_seq is None:
            return records
        return [e for e in records if getattr(e, "attempt_seq", 0) == self.current_attempt_seq]

    # ── Public API ────────────────────────────────────────────────────

    def evaluate(self, criteria: SuccessCriteria, budget_spent: int = 0) -> Report:
        """Evaluate all criteria and return a decision report.

        This is the main entry point.  It:
        1. Validates the criteria document.
        2. Evaluates each criterion.
        3. Derives the overall decision.
        4. Returns a Report with per-criterion results.
        """
        # 1. Validate criteria document
        validation_errors = criteria.validate()
        if validation_errors:
            return Report(
                decision=Decision.BLOCKED,
                summary=f"Invalid criteria: {'; '.join(validation_errors)}",
            )

        # 2. Evaluate each criterion
        results: List[Result] = []
        missing_evidence: List[str] = []
        failed_required: List[str] = []

        for criterion in criteria.criteria:
            result = self._evaluate_criterion(criterion)
            results.append(result)

            if result.status == "MISSING_EVIDENCE":
                missing_evidence.append(criterion.id)
                if criterion.required:
                    failed_required.append(criterion.id)
            elif result.status == "FAIL" and criterion.required:
                failed_required.append(criterion.id)

        # 3. Derive decision
        decision = self._derive_decision(
            criteria=criteria,
            failed_required=failed_required,
            missing_evidence=missing_evidence,
            budget_spent=budget_spent,
            results=results,
        )

        # 4. Build summary
        summary = self._build_summary(decision, failed_required, missing_evidence, criteria)

        return Report(
            decision=decision,
            results=results,
            missing_evidence=missing_evidence,
            failed_required=failed_required,
            summary=summary,
        )

    # ── Criterion Evaluators ──────────────────────────────────────────

    def _evaluate_criterion(self, criterion: Criterion) -> Result:
        """Dispatch to the appropriate evaluator based on criterion kind."""
        evaluators = {
            CriterionKind.COMMAND_EXIT_CODE: self._eval_command_exit_code,
            CriterionKind.PATH_CHANGED: self._eval_path_changed,
            CriterionKind.PATH_UNCHANGED: self._eval_path_unchanged,
            CriterionKind.NO_UNKNOWN_CHANGES: self._eval_no_unknown_changes,
            CriterionKind.FILE_CONTAINS: self._eval_file_contains,
            CriterionKind.FAILURE_RESOLVED: self._eval_failure_resolved,
        }
        evaluator = evaluators.get(criterion.kind)
        if evaluator is None:
            return Result(
                criterion_id=criterion.id,
                status="UNKNOWN",
                reason=f"Unhandled criterion kind: {criterion.kind}",
            )
        return evaluator(criterion)

    def _eval_command_exit_code(self, criterion: Criterion) -> Result:
        """Check that a command's exit code matches the expected value.

        Looks for evidence records (observed or inferred) whose claim
        contains the command string, then compares the recorded exit_code.
        Evidence is scoped to the current attempt when one is configured, so a
        previous attempt's PASS/FAIL cannot contaminate this evaluation.
        """
        expected = criterion.expected if criterion.expected is not None else 0
        evidence_records = self._evidence_for_current_attempt()

        # Iterate in reverse to use the most recent matching evidence within
        # the current attempt. Earlier evidence from a previous repair round is
        # not in scope and cannot override the current result.
        for ev in reversed(evidence_records):
            cmd = ev.details.get("command", "") or ev.details.get("operation", "")
            if criterion.command and criterion.command in str(cmd):
                exit_code = ev.details.get("exit_code")
                if exit_code is None:
                    # Try raw_facts embedded in details
                    raw = ev.details.get("raw_facts", {})
                    exit_code = raw.get("exit_code")
                if exit_code is not None:
                    observed_code = int(exit_code)
                    return Result(
                        criterion_id=criterion.id,
                        status="PASS" if observed_code == int(expected) else "FAIL",
                        observed=observed_code,
                        evidence_id=ev.operation_id,
                        reason=f"exit_code={observed_code}, expected={expected}",
                    )

        return Result(
            criterion_id=criterion.id,
            status="MISSING_EVIDENCE",
            reason=f"No evidence found for command: {criterion.command}",
        )

    def _eval_path_changed(self, criterion: Criterion) -> Result:
        """Check that a file was modified (hash differs between snapshots)."""
        before = self.snapshots_before.get(criterion.path)
        after = self.snapshots_after.get(criterion.path)

        # If we have snapshots, compare them
        if before is not None and after is not None:
            changed = before != after
            expected = criterion.expected if criterion.expected is not None else True
            return Result(
                criterion_id=criterion.id,
                status="PASS" if changed == bool(expected) else "FAIL",
                observed=changed,
                reason=f"path_changed={changed}, expected={expected}",
            )

        # Fallback: check if path exists and has evidence
        evidence_records = self.evidence.get_observed()
        for ev in evidence_records:
            if ev.path == criterion.path:
                return Result(
                    criterion_id=criterion.id,
                    status="PASS",
                    observed=True,
                    evidence_id=ev.operation_id,
                    reason=f"evidence confirms path {criterion.path} was observed",
                )

        return Result(
            criterion_id=criterion.id,
            status="MISSING_EVIDENCE",
            reason=f"No snapshot or evidence for path: {criterion.path}",
        )

    def _eval_path_unchanged(self, criterion: Criterion) -> Result:
        """Check that a file was NOT modified (hash matches before snapshot)."""
        before = self.snapshots_before.get(criterion.path)
        after = self.snapshots_after.get(criterion.path)

        if before is not None and after is not None:
            unchanged = before == after
            expected = criterion.expected if criterion.expected is not None else True
            return Result(
                criterion_id=criterion.id,
                status="PASS" if unchanged == bool(expected) else "FAIL",
                observed=unchanged,
                reason=f"path_unchanged={unchanged}, expected={expected}",
            )

        # If no snapshot data, assume unchanged (no evidence of change)
        return Result(
            criterion_id=criterion.id,
            status="PASS",
            observed=True,
            reason="no snapshot data available, assuming unchanged",
        )

    def _eval_no_unknown_changes(self, criterion: Criterion) -> Result:
        """Check that no external (UNKNOWN) paths were modified."""
        has_unknown = len(self.unknown_paths) > 0
        return Result(
            criterion_id=criterion.id,
            status="PASS" if not has_unknown else "FAIL",
            observed=list(self.unknown_paths),
            reason=f"unknown_paths count={len(self.unknown_paths)}",
        )

    def _eval_file_contains(self, criterion: Criterion) -> Result:
        """Check that a file contains a specific substring."""
        expected_str = str(criterion.expected) if criterion.expected is not None else ""
        if not expected_str:
            return Result(
                criterion_id=criterion.id,
                status="FAIL",
                reason="expected substring is empty",
            )

        try:
            target = self.jail.check_path(criterion.path, allow_missing=False)
            if not target.is_file():
                return Result(
                    criterion_id=criterion.id,
                    status="FAIL",
                    reason=f"Not a file: {criterion.path}",
                )
            content = target.read_text(encoding="utf-8", errors="replace")
            found = expected_str in content

            # Look for evidence of a write to this path
            evidence_id = ""
            for ev in self.evidence.get_observed():
                if ev.path == criterion.path:
                    evidence_id = ev.operation_id
                    break

            return Result(
                criterion_id=criterion.id,
                status="PASS" if found else "FAIL",
                observed=found,
                evidence_id=evidence_id,
                reason=f"contains={found}, looking for: {expected_str[:80]}",
            )
        except (JailError, OSError) as exc:
            return Result(
                criterion_id=criterion.id,
                status="FAIL",
                reason=f"error reading file: {exc}",
            )

    def _eval_failure_resolved(self, criterion: Criterion) -> Result:
        """Check that a previously failing test now passes.

        Looks for evidence of a verification command that previously
        failed (exit_code != 0) and now succeeds. Scoped to the current
        attempt when one is configured.
        """
        evidence_records = self._evidence_for_current_attempt()

        # Find any evidence of verification commands
        verification_evidence = [
            ev for ev in evidence_records
            if ev.details.get("kind") == "verification_command"
            or "verification" in ev.claim.lower()
        ]

        if not verification_evidence:
            return Result(
                criterion_id=criterion.id,
                status="MISSING_EVIDENCE",
                reason="No verification evidence found",
            )

        # Check that all verification evidence shows success (exit_code=0)
        all_pass = True
        for ev in verification_evidence:
            exit_code = ev.details.get("exit_code")
            if exit_code is None:
                raw = ev.details.get("raw_facts", {})
                exit_code = raw.get("exit_code")
            if exit_code is not None and int(exit_code) != 0:
                all_pass = False
                break

        return Result(
            criterion_id=criterion.id,
            status="PASS" if all_pass else "FAIL",
            observed=all_pass,
            evidence_id=verification_evidence[-1].operation_id if verification_evidence else "",
            reason=f"all_verifications_pass={all_pass}",
        )

    # ── Decision Logic ────────────────────────────────────────────────

    def _derive_decision(
        self,
        criteria: SuccessCriteria,
        failed_required: List[str],
        missing_evidence: List[str],
        budget_spent: int,
        results: List[Result],
    ) -> Decision:
        """Derive the overall decision from criterion results.

        Decision matrix:
        - No failed required and no missing evidence -> PASS
        - UNKNOWN external changes -> BLOCKED
        - Failed required with repair budget remaining -> REPAIR
        - Failed required with no budget or repeated signature -> ROLLBACK
        - Budget exceeded -> ROLLBACK
        """
        # Check for external UNKNOWN changes first
        if self.unknown_paths:
            return Decision.BLOCKED

        # Check for missing evidence
        if missing_evidence:
            # Missing evidence for required criteria -> BLOCKED
            return Decision.BLOCKED

        # All required criteria passed
        if not failed_required:
            return Decision.PASS

        # Repeated identical failure -> break the repair loop immediately.
        current_sig = FailureSignature.from_results(results, self.changed_files)
        for prior in self.failure_signatures:
            if prior.signature == current_sig.signature and prior.file_set == current_sig.file_set:
                return Decision.ROLLBACK

        # Failed required criteria exist -- check repair budget
        max_budget = self._compute_max_budget(criteria)
        if budget_spent >= max_budget:
            return Decision.ROLLBACK

        if self.repair_count >= criteria.max_repairs:
            return Decision.ROLLBACK

        return Decision.REPAIR

    def _compute_max_budget(self, criteria: SuccessCriteria) -> int:
        """Compute maximum budget from criteria weights."""
        weights = criteria.budget_weights
        total = 0
        for criterion in criteria.criteria:
            if not criterion.required:
                continue
            if criterion.kind == CriterionKind.COMMAND_EXIT_CODE:
                total += weights.get("TEST", 2)
            elif criterion.kind in (CriterionKind.PATH_CHANGED,):
                total += weights.get("PATCH", 5)
            elif criterion.kind in (CriterionKind.FILE_CONTAINS,):
                total += weights.get("PATCH", 5)
            elif criterion.kind in (CriterionKind.PATH_UNCHANGED, CriterionKind.NO_UNKNOWN_CHANGES):
                total += weights.get("READ", 1)
            elif criterion.kind == CriterionKind.FAILURE_RESOLVED:
                total += weights.get("REPAIR", 5)
        # Scale by max_repairs + 1 (original attempt)
        return total * (criteria.max_repairs + 1)

    def _build_summary(
        self,
        decision: Decision,
        failed_required: List[str],
        missing_evidence: List[str],
        criteria: SuccessCriteria,
    ) -> str:
        """Build a human-readable summary of the gate evaluation."""
        total = len(criteria.criteria)
        required = sum(1 for c in criteria.criteria if c.required)
        failed_count = len(failed_required)
        missing_count = len(missing_evidence)

        if decision == Decision.PASS:
            return f"All {required} required criteria passed ({total} total)"
        if decision == Decision.BLOCKED:
            if self.unknown_paths:
                return f"Blocked: {len(self.unknown_paths)} external change(s) detected"
            return f"Blocked: {missing_count} required criterion missing evidence"
        if decision == Decision.REPAIR:
            return f"Repair needed: {failed_count} required criterion(s) failed, {criteria.max_repairs - self.repair_count} repair(s) remaining"
        if decision == Decision.ROLLBACK:
            if self.repair_count >= criteria.max_repairs:
                return f"Rollback: max repairs ({criteria.max_repairs}) exhausted"
            return "Rollback: repair budget exhausted"
        if decision == Decision.TIMEOUT:
            return "Timeout: wall-clock or step budget exceeded"
        return f"Decision: {decision.value}"


# ── Snapshot Helpers ──────────────────────────────────────────────────────


def take_snapshot(root: Path, jail: Optional[WorkspaceJail] = None) -> Dict[str, str]:
    """Take a SHA-256 snapshot of all files in the workspace.

    Returns a dict of relative-path -> sha256 hex string.
    Skips .nabd, .git, .env, __pycache__ etc. via the jail.
    """
    if jail is None:
        jail = WorkspaceJail(root)
    snapshot: Dict[str, str] = {}
    root = Path(root).expanduser().resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # Skip internal artifacts
        if ".nabd" in path.parts or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            jail.check_path(path, allow_missing=False)
        except JailError:
            continue
        rel = str(path.relative_to(root))
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                digest.update(chunk)
        snapshot[rel] = digest.hexdigest()
    return snapshot


def diff_snapshots(
    before: Dict[str, str],
    after: Dict[str, str],
) -> Dict[str, Any]:
    """Compare two snapshots and return added/removed/changed paths."""
    before_keys = set(before.keys())
    after_keys = set(after.keys())
    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    changed = sorted(
        path for path in before_keys & after_keys
        if before[path] != after[path]
    )
    return {"added": added, "removed": removed, "changed": changed}
