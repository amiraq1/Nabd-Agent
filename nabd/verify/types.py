"""Type definitions for the Verification Gate.

Every criterion is a deterministic, falsifiable check.  The gate never
issues PASS based on LLM text alone -- every result must originate from
an evidence-backed evaluation recorded in the EvidenceStore.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CriterionKind(str, Enum):
    """Recognised criterion types.

    Only these kinds are permitted in a SuccessCriteria document.
    Any unknown kind causes the gate to reject the criteria.
    """

    COMMAND_EXIT_CODE = "command_exit_code"
    PATH_CHANGED = "path_changed"
    PATH_UNCHANGED = "path_unchanged"
    NO_UNKNOWN_CHANGES = "no_unknown_changes"
    FILE_CONTAINS = "file_contains"
    FAILURE_RESOLVED = "failure_resolved"


class Decision(str, Enum):
    """Outcome of a verification gate evaluation.

    | Decision   | Meaning                                       |
    |------------|-----------------------------------------------|
    | PASS       | All mandatory criteria met with evidence       |
    | REPAIR     | Failure is repairable and budget remains       |
    | ROLLBACK   | Irreparable failure or repeated no-improvement |
    | BLOCKED    | External UNKNOWN change or missing evidence    |
    | TIMEOUT    | Wall-clock or step budget exhausted            |
    """

    PASS = "PASS"
    REPAIR = "REPAIR"
    ROLLBACK = "ROLLBACK"
    BLOCKED = "BLOCKED"
    TIMEOUT = "TIMEOUT"


# ── Criterion ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Criterion:
    """A single success criterion declared in a SuccessCriteria document.

    Fields mirror the JSON schema in the plan.  ``expected`` is typed as
    ``Any`` because its semantics depend on ``kind``:

    * ``command_exit_code``  -> ``int`` (default 0)
    * ``path_changed``      -> ``bool`` (default True)
    * ``path_unchanged``    -> ``bool`` (default True)
    * ``file_contains``     -> ``str`` (substring to find)
    * ``failure_resolved``  -> ``bool`` (default True)

    ``no_unknown_changes`` ignores ``expected`` (always checks for empty
    unknown-paths set).
    """

    id: str
    kind: CriterionKind
    required: bool = True
    path: str = ""
    command: str = ""
    expected: Any = None


# ── Success Criteria Document ────────────────────────────────────────────


@dataclass
class SuccessCriteria:
    """Top-level contract describing a task and its success conditions.

    The gate *validates* the criteria document before evaluation: unknown
    criterion kinds are rejected, required fields are enforced per kind,
    and ``max_repairs`` / ``wall_clock_timeout_seconds`` must be positive.
    """

    task_id: str
    description: str
    criteria: List[Criterion] = field(default_factory=list)
    max_repairs: int = 3
    wall_clock_timeout_seconds: int = 300
    budget_weights: Dict[str, int] = field(
        default_factory=lambda: {
            "READ": 1,
            "SEARCH": 1,
            "TEST": 2,
            "PATCH": 5,
            "REPAIR": 5,
        }
    )

    # ── validation ────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Return a list of validation error strings; empty means valid."""
        errors: List[str] = []
        if not self.task_id:
            errors.append("task_id is required")
        if not self.description:
            errors.append("description is required")
        if self.max_repairs < 0:
            errors.append("max_repairs must be >= 0")
        if self.wall_clock_timeout_seconds <= 0:
            errors.append("wall_clock_timeout_seconds must be > 0")
        seen_ids: set[str] = set()
        for criterion in self.criteria:
            if criterion.id in seen_ids:
                errors.append(f"duplicate criterion id: {criterion.id}")
            seen_ids.add(criterion.id)
            errors.extend(self._validate_criterion(criterion))
        return errors

    @staticmethod
    def _validate_criterion(c: Criterion) -> List[str]:
        errors: List[str] = []
        if not c.id:
            errors.append("criterion id is required")
        if c.kind not in CriterionKind:
            errors.append(f"unknown criterion kind: {c.kind}")
        if c.kind == CriterionKind.COMMAND_EXIT_CODE and not c.command:
            errors.append(f"criterion {c.id}: command is required for command_exit_code")
        if c.kind in (CriterionKind.PATH_CHANGED, CriterionKind.PATH_UNCHANGED, CriterionKind.FILE_CONTAINS) and not c.path:
            errors.append(f"criterion {c.id}: path is required for {c.kind.value}")
        if c.kind == CriterionKind.FILE_CONTAINS and c.expected is None:
            errors.append(f"criterion {c.id}: expected (substring) is required for file_contains")
        return errors


# ── Result ────────────────────────────────────────────────────────────────


@dataclass
class Result:
    """Evaluation outcome for a single criterion.

    ``observed`` holds the actual value seen during evaluation (e.g. exit
    code, hash diff, boolean).  ``evidence_id`` links back to the
    EvidenceStore record that proves the observation.
    """

    criterion_id: str
    status: str  # "PASS" | "FAIL" | "MISSING_EVIDENCE" | "UNKNOWN"
    observed: Any = None
    evidence_id: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Report ────────────────────────────────────────────────────────────────


@dataclass
class Report:
    """Aggregated gate report after evaluating all criteria.

    ``decision`` is the single FSM-compatible outcome.  ``results`` is
    the per-criterion detail.  ``failed_required`` lists IDs of mandatory
    criteria that did not pass.  ``missing_evidence`` lists IDs where no
    evidence record was found at all.
    """

    decision: Decision
    results: List[Result] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    failed_required: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "results": [r.to_dict() for r in self.results],
            "missing_evidence": list(self.missing_evidence),
            "failed_required": list(self.failed_required),
            "summary": self.summary,
        }


# ── Failure Signature ────────────────────────────────────────────────────


@dataclass
class FailureSignature:
    """Compact fingerprint of a verification failure for streak detection.

    Two failures are considered *the same* when ``signature`` matches
    exactly.  ``file_set`` tracks which files were touched; a matching
    signature with the same file set and no improvement count signals
    that repair is stuck.
    """

    signature: str = ""
    file_set: List[str] = field(default_factory=list)
    no_improvement_streak: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_report(report: Report, changed_files: Optional[List[str]] = None) -> "FailureSignature":
        """Derive a signature from a gate report.

        The signature is a deterministic string built from failed required
        criteria IDs and their observed values, so identical failures
        produce identical signatures.
        """
        parts: List[str] = []
        for result in report.results:
            if result.status == "FAIL":
                parts.append(f"{result.criterion_id}:{result.observed}")
        sig = "|".join(sorted(parts))
        return FailureSignature(
            signature=sig,
            file_set=sorted(changed_files or []),
        )
