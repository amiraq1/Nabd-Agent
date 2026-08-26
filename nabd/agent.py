"""Nabd coding-agent orchestration loop."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

from .approval import ApprovalMode, ApprovalProvider, CallbackApprovalProvider, InteractiveApprovalProvider, coerce_approval_mode
from .evidence import EvidenceStore
from .fsm import FSM, FSMError, State
from .llm import LLMClient, LLMError
from .models import AgentResult, Plan, ToolCall, ToolResult
from .tools import ToolExecutor, format_call
from .verify.gate import VerificationGate, diff_snapshots, take_snapshot
from .reconcile import typed_snapshot, classify
from .verify.types import (
    Criterion,
    CriterionKind,
    Decision,
    FailureSignature,
    Report,
    SuccessCriteria,
)
from .events import AgentEventSink, EventType, build_event

SYSTEM_PROMPT = """You are Nabd, a senior software engineer operating inside a user's local project.
You must work in small, verifiable steps and return ONLY valid JSON.
Never assume a file exists: inspect it first. Never use paths outside the selected project root.
Prefer minimal, maintainable changes. Preserve existing behavior unless the task asks otherwise.
Do not expose or request secrets. Do not read .env files, git internals, SSH keys, or credentials.
For destructive commands, package installs, network access, database changes, or deleting files,
ask the user for confirmation by describing the action in the plan; the terminal wrapper may reject it.

Your JSON must have exactly this shape:
{
  "summary": "short plan or result",
  "steps": ["ordered human-readable steps"],
  "actions": [
    {"name":"list_files|read_file|write_file|search|run_command", "arguments": {}}
  ],
  "verification": ["commands to verify the changes, e.g. pytest -q or python -m compileall ."]
}
Only use the five listed tools. The search tool accepts {"query": "text", "path": ".", "max_results": 50}. write_file must include complete file content, not a patch.
"""


# Tool names the model may hallucinate as a verification shell command.
# They are agent tools, not shell commands, so executing them only produces
# spurious failures -- strip them before building run_command calls.
BLOCKED_VERIFICATION = frozenset(
    {
        "run_command",
        "write_file",
        "read_file",
        "list_files",
        "search",
        "write",
        "read",
        "list",
        "shell",
    }
)


READ_ONLY_KEYWORDS = frozenset(
    {
        "افحص",
        "فحص",
        "ملخص",
        "تلخيص",
        "تقرير",
        "اقرأ",
        "قراءة",
        "جرد",
        "audit",
        "inspect",
        "summarize",
        "summary",
        "report",
        "review",
        "list",
    }
)

MUTATING_KEYWORDS = frozenset(
    {
        "أنشئ",
        "انشئ",
        "اكتب",
        "عدّل",
        "عدل",
        "أصلح",
        "اصلح",
        "احذف",
        "حذف",
        "غيّر",
        "غير",
        "create",
        "write",
        "edit",
        "fix",
        "refactor",
        "modify",
        "delete",
        "remove",
    }
)


# Negation markers that turn a mutating verb into a read-only intent
# (e.g. "لا تعدّل" = "do NOT modify", "don't edit", "do not delete").
_NEGATION_MARKERS = ("لا", "لا ت", "لا ي", "don't", "do not", "cannot", "won't", "never")


def _is_negated(text: str, keyword: str) -> bool:
    """True if *keyword* appears immediately after a negation marker."""
    index = text.find(keyword)
    if index <= 0:
        return False
    head = text[:index].rstrip()
    return any(head.endswith(marker) for marker in _NEGATION_MARKERS)


def classify_intent(task: str) -> str:
    """Classify task scope before planning; unknown tasks remain mutating.

    A mutating keyword only forces MUTATING when it is not negated. A command
    such as "لا تعدّل أي ملف" (do NOT modify any file) must stay READ_ONLY even
    though it contains the mutating verb "عدّل".
    """
    text = " ".join(str(task).lower().split())
    mutating_hits = [kw for kw in MUTATING_KEYWORDS if kw in text]
    if mutating_hits and not all(_is_negated(text, kw) for kw in mutating_hits):
        return "MUTATING"
    if any(keyword in text for keyword in READ_ONLY_KEYWORDS):
        return "READ_ONLY"
    if mutating_hits:
        # The only mutating signal was a negated verb (e.g. "don't edit") ->
        # the user explicitly wants no mutation, so treat as READ_ONLY.
        return "READ_ONLY"
    return "MUTATING"


def _sanitize_verification(commands: Any) -> List[str]:
    """Strip hallucinated tool-name prefixes from the verification list."""
    if not isinstance(commands, list):
        return []
    prefix_re = re.compile(
        r"(?:" + "|".join(re.escape(name) for name in sorted(BLOCKED_VERIFICATION, key=len, reverse=True)) + r")\b"
    )
    cleaned: List[str] = []
    for raw in commands:
        command = str(raw).strip()
        match = prefix_re.match(command)
        if match:
            command = command[match.end():].lstrip(": ").strip()
        if command and command not in BLOCKED_VERIFICATION:
            cleaned.append(command)
    return cleaned[:8]


def _as_plan(data: Dict[str, Any]) -> Plan:
    raw_actions = data.get("actions", [])
    actions: List[ToolCall] = []
    if not isinstance(raw_actions, list):
        raise LLMError("Model field actions must be a list")
    for item in raw_actions[:30]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise LLMError("Each action must contain a tool name")
        arguments = item.get("arguments", {})
        if not isinstance(arguments, dict):
            raise LLMError("Tool arguments must be an object")
        actions.append(ToolCall(item["name"], arguments))

    verification = _sanitize_verification(data.get("verification"))
    return Plan(
        summary=str(data.get("summary", "")),
        steps=[str(step) for step in data.get("steps", [])][:20],
        tool_calls=actions,
        verification=verification,
    )


def _context(task: str, files: str, history: List[ToolResult], failures: List[ToolResult]) -> str:
    payload = {
        "task": task,
        "project_files": files,
        "recent_tool_results": [result.as_dict() for result in history[-12:]],
        "verification_failures": [result.as_dict() for result in failures[-8:]],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_success_criteria(
    task_id: str,
    task_description: str,
    verification_commands: List[str],
    changed_files: List[str],
) -> SuccessCriteria:
    """Build a SuccessCriteria document from the plan's verification commands.

    This is the bridge between the LLM-proposed verification and the
    deterministic gate.  The model proposes commands; the gate evaluates
    them against evidence.
    """
    criteria: List[Criterion] = []

    # 1. Each verification command becomes a command_exit_code criterion
    for index, command in enumerate(verification_commands):
        criteria.append(
            Criterion(
                id=f"verify_{index}",
                kind=CriterionKind.COMMAND_EXIT_CODE,
                command=command,
                expected=0,
                required=True,
            )
        )

    # 2. At least one changed file must exist (if any files were written)
    if changed_files:
        criteria.append(
            Criterion(
                id="target_file_changed",
                kind=CriterionKind.NO_UNKNOWN_CHANGES,
                required=True,
            )
        )

    # 3. No external changes allowed
    criteria.append(
        Criterion(
            id="no_unknown_changes",
            kind=CriterionKind.NO_UNKNOWN_CHANGES,
            required=True,
        )
    )

    return SuccessCriteria(
        task_id=task_id,
        description=task_description,
        criteria=criteria,
        max_repairs=3,
        wall_clock_timeout_seconds=300,
    )


class NabdAgent:
    def __init__(
        self,
        root: Path,
        provider: str = "auto",
        auto_approve: bool = False,
        workspace_free: bool = False,
        llm_client: Optional[LLMClient] = None,
        controlled_mutation: bool = False,
        event_sink: Optional[AgentEventSink] = None,
        approval_callback: Optional[Callable[[Mapping[str, Any]], bool]] = None,
        approval_mode: ApprovalMode | str | None = None,
        approval_provider: Optional[ApprovalProvider] = None,
    ) -> None:
        # Test-only dependency injection: a fake client may be supplied so the
        # agent can run deterministically offline. In production llm_client is
        # None and the real LLMClient(provider) is used exactly as before.
        self.client = llm_client if llm_client is not None else LLMClient(provider)
        self.fsm = FSM()
        self.history: List[ToolResult] = []
        # The public runtime requires approval_mode explicitly. The lower-level
        # constructor retains a compatibility bridge for existing CLI/tests:
        # legacy auto_approve/workspace_free opt into AUTO; otherwise CONFIRM.
        self.approval_mode = (
            coerce_approval_mode(approval_mode)
            if approval_mode is not None
            else (ApprovalMode.AUTO if (auto_approve or workspace_free) else ApprovalMode.CONFIRM)
        )
        if workspace_free and self.approval_mode is not ApprovalMode.AUTO:
            raise ValueError("workspace_free requires explicit approval_mode=AUTO")
        if auto_approve and self.approval_mode is not ApprovalMode.AUTO:
            raise ValueError("auto_approve requires explicit approval_mode=AUTO")
        self.auto_approve = self.approval_mode is ApprovalMode.AUTO
        self.workspace_free = workspace_free
        if approval_provider is not None:
            self.approval_provider = approval_provider
        elif approval_callback is not None:
            self.approval_provider = CallbackApprovalProvider(approval_callback)
        else:
            self.approval_provider = InteractiveApprovalProvider()
        self.intent = "MUTATING"
        self.task_id = EvidenceStore.new_task_id()
        self.evidence = EvidenceStore(root, task_id=self.task_id)
        self.executor = ToolExecutor(
            root,
            approve=self._approve,
            auto_approve=self.auto_approve,
            evidence_store=self.evidence,
            controlled_mutation=controlled_mutation,
        )
        # M1 Controlled Mutation: bootstrap the kernel marker so the agent's
        # mutating tools have a valid contract to gate against.
        if controlled_mutation and self.executor.controlled is not None:
            try:
                self.executor.controlled.ensure_marker()
            except OSError:
                pass
        # Verification Gate state
        self._snapshot_before: Dict[str, str] = {}
        self._snapshot_after: Dict[str, str] = {}
        self._snapshot_invalid: bool = False
        self._unknown_paths: Set[str] = set()
        # M3 UNKNOWN reconciliation state.
        self._snapshot_before_typed: Dict[str, Dict[str, Any]] = {}
        self._change_classification: List[Dict[str, Any]] = []
        self._changed_files: List[str] = []
        self._repair_count: int = 0
        self._budget_spent: int = 0
        # M6: history of failure signatures across repair attempts, used by the
        # gate to break repeated-identical-failure (infinite repair) loops.
        self._failure_signatures: List[FailureSignature] = []
        # M6: per-attempt sequence so the gate evaluates only the current
        # attempt's evidence and old PASS/FAIL records cannot contaminate it.
        self._attempt_seq: int = 0
        self._start_time: float = 0.0
        # M7/M8 (UI-4): optional event stream + approval callback. Both are
        # additive: core behaviour is identical when they are None.
        self.event_sink: Optional[AgentEventSink] = event_sink
        self.session_id: str = uuid.uuid4().hex
        self._event_seq: int = 0

    def _publish(
        self,
        event_type: Any,
        attempt_id: Optional[str] = None,
        evidence_id: Optional[str] = None,
        payload: Any = None,
        **kwargs: Any,
    ) -> None:
        """Publish a single UiEvent to the optional sink (no-op when absent).

        A broken UI sink can never crash the agent or bypass a safety policy:
        any exception from the sink is swallowed here.
        """
        sink = self.event_sink
        if sink is None:
            return
        self._event_seq += 1
        aid = attempt_id if attempt_id is not None else str(self._attempt_seq)
        merged: Dict[str, Any] = {}
        if isinstance(payload, dict):
            merged.update(payload)
        merged.update(kwargs)
        event = build_event(
            event_type,
            seq=self._event_seq,
            attempt_order=self._attempt_seq,
            task_id=self.task_id,
            session_id=self.session_id,
            attempt_id=aid,
            fsm_state=self.fsm.state.name,
            evidence_id=evidence_id,
            source="core",
            payload=merged,
        )
        try:
            sink.publish(event)
        except Exception:
            pass

    def _approve(self, call: ToolCall) -> bool:
        request = {
            "tool": call.name,
            "arguments": call.arguments,
            "display": format_call(call),
        }
        self._publish(EventType.APPROVAL_REQUIRED, payload={"request": request})
        if self.approval_mode is ApprovalMode.AUTO:
            self._publish(
                EventType.APPROVAL_ACCEPTED,
                payload={"request": request, "auto": True},
            )
            return True
        # CONFIRM is core-owned and synchronous. The provider may be a Rich/UI
        # adapter, but it never receives authority to bypass ToolExecutor policy.
        decision = self.approval_provider.decide(request)
        self._publish(
            EventType.APPROVAL_ACCEPTED if decision else EventType.APPROVAL_DENIED,
            payload={"request": request, "decision": decision},
        )
        return decision

    def _run_calls(self, calls: List[ToolCall]) -> List[ToolResult]:
        results: List[ToolResult] = []
        for call in calls:
            self._publish(
                EventType.TOOL_STARTED,
                payload={"tool": call.name, "arguments": call.arguments},
            )
            if call.name == "write_file":
                self._publish(
                    EventType.MUTATION_STARTED,
                    payload={"path": call.arguments.get("path")},
                )
            print(f"\n[أداة] {format_call(call)}")
            result = self.executor.execute(call)
            self.history.append(result)
            results.append(result)
            marker = "نجاح" if result.ok else "فشل"
            print(f"[{marker}] {result.output[:4000]}")
            if result.ok:
                self._publish(
                    EventType.TOOL_SUCCEEDED,
                    payload={"tool": call.name, "output_excerpt": result.output[:4000]},
                )
            else:
                self._publish(
                    EventType.TOOL_FAILED,
                    payload={"tool": call.name, "error": result.output[:4000]},
                )
        return results

    def _ask(self, task: str, files: str, failures: List[ToolResult]) -> Plan:
        response = self.client.complete_json(
            SYSTEM_PROMPT,
            _context(task, files, self.history, failures),
        )
        return _as_plan(response)

    def _take_snapshot_before(self) -> None:
        """Take a filesystem snapshot before task execution begins.

        Implements the M0.5 (Snapshot Integrity Precondition) gate: if the
        snapshot cannot be taken, or is incomplete/corrupt, the agent MUST NOT
        proceed. A valid snapshot is also persisted so a subsequent run can
        detect and reject a stale/corrupt manifest and rebuild cleanly.
        """
        self._snapshot_invalid = False
        root = Path(self.executor.root).expanduser().resolve()
        manifest_path = root / ".nabd" / "snapshot-before.json"

        # Restart safety: a stale/corrupt persisted manifest from a previous
        # interrupted run must be rejected and rebuilt from scratch.
        if manifest_path.exists() and not self._manifest_is_valid(manifest_path, root):
            try:
                manifest_path.unlink()
            except OSError as exc:
                self._snapshot_invalid = True
                self._snapshot_before = {}
                self._snapshot_error = f"cannot remove stale snapshot: {exc}"
                return

        try:
            snap = take_snapshot(root)
        except Exception as exc:
            self._snapshot_before = {}
            self._snapshot_invalid = True
            self._snapshot_error = f"snapshot capture failed: {exc}"
            return

        expected = self._expected_snapshot_keys(root)
        if (
            set(snap.keys()) != expected
            or any(not isinstance(value, str) or len(value) != 64 for value in snap.values())
        ):
            # Incomplete/corrupt snapshot: refuse before any tool can run.
            self._snapshot_before = snap
            self._snapshot_invalid = True
            self._snapshot_error = "snapshot manifest is incomplete or malformed"
            return

        self._snapshot_before = snap
        # M3: keep a typed (symlink-aware) snapshot for UNKNOWN reconciliation.
        try:
            self._snapshot_before_typed = typed_snapshot(self.executor.root)
        except Exception as exc:
            self._snapshot_invalid = True
            self._snapshot_error = f"typed snapshot capture failed: {exc}"
            return

        # Persist with temp-file -> fsync -> replace -> directory fsync.
        temporary = manifest_path.with_name(
            f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps({"files": snap}, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, manifest_path)
            directory_fd = os.open(manifest_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (OSError, TypeError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            self._snapshot_invalid = True
            self._snapshot_error = f"snapshot manifest persistence failed: {exc}"
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _expected_snapshot_keys(root: Path) -> Set[str]:
        """Files take_snapshot would cover (mirrors its .nabd/.git skip rules)."""
        root = Path(root).expanduser().resolve()
        keys: Set[str] = set()
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if ".nabd" in path.parts or ".git" in path.parts or "__pycache__" in path.parts:
                continue
            keys.add(str(path.relative_to(root)))
        return keys

    @staticmethod
    def _manifest_is_valid(path: Path, root: Path) -> bool:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return False
        files = data.get("files") if isinstance(data, dict) else None
        if not isinstance(files, dict):
            return False
        try:
            current = take_snapshot(root)
        except Exception:
            return False
        return files == current

    def _take_snapshot_after(self) -> None:
        """Take a filesystem snapshot after execution and detect unknown changes."""
        try:
            self._snapshot_after = take_snapshot(self.executor.root)
            diff = diff_snapshots(self._snapshot_before, self._snapshot_after)
            # Unknown paths = changes not in our changed_files list
            all_changed = set(diff["added"] + diff["removed"] + diff["changed"])
            known = set(self._changed_files)
            self._unknown_paths = all_changed - known
            # M3: typed (symlink-safe) reconciliation. This classifies every
            # delta as AGENT_* or UNKNOWN_* and surfaces injected symlinks that
            # point outside the workspace. UNKNOWN paths are never rolled back.
            after_typed = typed_snapshot(self.executor.root)
            self._change_classification = classify(
                self._snapshot_before_typed, after_typed, known
            )
            unknown_from_recon = {
                r["path"]
                for r in self._change_classification
                if r["category"].startswith("UNKNOWN")
            }
            self._unknown_paths = self._unknown_paths | unknown_from_recon
        except Exception:
            self._snapshot_after = {}

    def _check_timeout(self) -> bool:
        """Return True if the wall-clock timeout has been exceeded."""
        if self._start_time <= 0:
            return False
        elapsed = time.time() - self._start_time
        return elapsed > 300  # default 300s timeout

    def _record_verification_evidence(self, verification: List[ToolResult], plan: Plan) -> None:
        """Record verification results in the evidence store."""
        for command, result in zip(plan.verification, verification):
            if result.raw_facts is not None:
                self.evidence.verify(
                    result.raw_facts,
                    claim=f"verification: {command}",
                    task_id=self.task_id,
                    relevant=result.ok,
                    attempt_seq=self._attempt_seq,
                )

    def _report_evidence_ids(self, report: Report) -> List[str]:
        """Return only evidence IDs already issued by the core EvidenceStore."""
        return [
            str(result.evidence_id)
            for result in report.results
            if getattr(result, "evidence_id", "")
        ]

    def _run_with_gate(
        self,
        task: str,
        plan: Plan,
        action_results: List[ToolResult],
        verification: List[ToolResult],
    ) -> Report:
        """Run the verification gate and return the report."""
        # Build success criteria from the plan
        criteria = _build_success_criteria(
            task_id=self.task_id,
            task_description=task,
            verification_commands=plan.verification,
            changed_files=self._changed_files,
        )

        # Take after-snapshot and detect unknown changes
        self._take_snapshot_after()

        # Create gate and evaluate
        gate = VerificationGate(
            root=self.executor.root,
            evidence=self.evidence,
            unknown_paths=self._unknown_paths,
            snapshots_before=self._snapshot_before,
            snapshots_after=self._snapshot_after,
            changed_files=self._changed_files,
            repair_count=self._repair_count,
            failure_signatures=self._failure_signatures,
            current_attempt_seq=self._attempt_seq,
        )

        report = gate.evaluate(criteria, budget_spent=self._budget_spent)

        # Record the decision in evidence
        self.evidence.add_inferred(
            claim="verification_decision",
            details=report.to_dict(),
            task_id=self.task_id,
        )

        return report

    def run(self, task: str, max_rounds: int = 5) -> AgentResult:
        try:
            self._start_time = time.time()

            # M0.5 Snapshot Integrity Precondition: this is deliberately the
            # first runtime operation. No intent transition, inventory, LLM,
            # plan, subprocess, or mutation is allowed before it succeeds.
            self._take_snapshot_before()
            if self._snapshot_invalid:
                self.evidence.add_inferred(
                    claim="pre-run snapshot integrity",
                    details={
                        "status": "REJECTED",
                        "reason": self._snapshot_error or "snapshot failed",
                    },
                )
                self.evidence.save()
                if self.fsm.can_transition(State.REJECTED):
                    self.fsm.transition(State.REJECTED)
                self._publish(
                    EventType.TASK_FAILED,
                    payload={
                        "summary": "Refusing to start: snapshot failed or incomplete (M0.5)",
                        "state": self.fsm.state.name,
                        "error": "M0.5_SNAPSHOT_INTEGRITY failure",
                    },
                )
                return AgentResult(
                    ok=False,
                    state=self.fsm.state.name,
                    summary="Refusing to start: snapshot failed or incomplete (M0.5)",
                    changes=[],
                    verification=[],
                    evidence=[item.to_dict() for item in self.evidence.get_all()],
                    error="M0.5_SNAPSHOT_INTEGRITY failure",
                )

            self.intent = "MUTATING" if self.workspace_free else classify_intent(task)
            self.executor.set_intent(self.intent)
            self._publish(EventType.TASK_ACCEPTED, payload={"task": task, "intent": self.intent})
            self._publish(EventType.INTENT_CLASSIFIED, payload={"intent": self.intent})
            inventory = self.executor.execute(ToolCall("list_files", {"path": "."}))
            self.history.append(inventory)
            files = inventory.output
            failures: List[ToolResult] = []
            last_summary = ""
            changes: List[str] = []
            self._publish(EventType.SNAPSHOT_READY, payload={"files": len(self._snapshot_before)})

            for round_number in range(1, max_rounds + 1):
                print(f"\n===== دورة الوكيل {round_number}/{max_rounds} =====")

                # Check wall-clock timeout
                if self._check_timeout():
                    print("\nانتهت مهلة المهمة؛ سينتقل إلى FAILED.")
                    self.evidence.save()
                    self._publish(EventType.TIMEOUT, payload={"scope": "wall_clock", "summary": last_summary})
                    if self.fsm.can_transition(State.FAILED):
                        self.fsm.transition(State.FAILED)
                    elif self.fsm.can_transition(State.REJECTED):
                        self.fsm.transition(State.REJECTED)
                    self._publish(EventType.TASK_FAILED, payload={"summary": last_summary, "state": self.fsm.state.name, "error": "Timeout: wall-clock budget exceeded"})
                    return AgentResult(
                        ok=False,
                        state=self.fsm.state.name,
                        summary=last_summary,
                        changes=changes,
                        verification=failures,
                        evidence=[item.to_dict() for item in self.evidence.get_all()],
                        error="Timeout: wall-clock budget exceeded",
                    )

                # M6: each loop iteration is a new attempt; scope evidence and
                # gate evaluation to this attempt sequence.
                self._attempt_seq = round_number

                if self.fsm.state == State.PLANNING:
                    plan = self._ask(task, files, failures)
                    last_summary = plan.summary
                    self._publish(EventType.PLAN_READY, payload={"summary": plan.summary, "steps": plan.steps, "actions": [c.name for c in plan.tool_calls], "verification": plan.verification})
                    print(f"\nالخطة: {plan.summary}")
                    for index, step in enumerate(plan.steps, 1):
                        print(f"  {index}. {step}")
                    self.fsm.transition(State.EXECUTING)
                else:
                    # After a failed verification the FSM is already EXECUTING;
                    # request a repair plan without a redundant self-transition.
                    plan = self._ask(task, files, failures)
                    last_summary = plan.summary
                    self._publish(EventType.PLAN_READY, payload={"summary": plan.summary, "steps": plan.steps, "actions": [c.name for c in plan.tool_calls], "verification": plan.verification})

                action_results = self._run_calls(plan.tool_calls)
                for call, result in zip(plan.tool_calls, action_results):
                    if result.raw_facts is not None:
                        self.evidence.verify(
                            result.raw_facts,
                            claim=f"tool {call.name}",
                            task_id=self.task_id,
                            relevant=call.name in {"write_file", "read_file", "search", "run_command"},
                            attempt_seq=self._attempt_seq,
                        )
                # Track changed files (associate each result with its own call,
                # so every mutated path is recorded exactly once).
                for call, result in zip(plan.tool_calls, action_results):
                    if result.ok and call.name == "write_file" and call.arguments.get("path"):
                        self._changed_files.append(str(call.arguments["path"]))
                changes.extend(result.output for result in action_results if result.ok and result.name == "write_file")
                policy_failures = [
                    result
                    for result in action_results
                    if result.raw_facts is not None
                    and result.raw_facts.status == "MUTATION_NOT_ALLOWED"
                ]
                self.fsm.transition(State.VERIFYING)
                if policy_failures:
                    failures = policy_failures
                    print("\nتم حجب تغيير خارج نطاق المهمة؛ سيعيد الوكيل التخطيط.")
                    self.fsm.transition(State.EXECUTING)
                    continue

                self._publish(EventType.VERIFICATION_STARTED, payload={"commands": plan.verification})
                verification_calls = [ToolCall("run_command", {"command": command}) for command in plan.verification]
                verification = self._run_calls(verification_calls)
                self._record_verification_evidence(verification, plan)

                # Use the Verification Gate for decision
                report = self._run_with_gate(task, plan, action_results, verification)

                print(f"\n[بوابة التحقق] القرار: {report.decision.value}")
                print(f"  {report.summary}")
                report_evidence_ids = self._report_evidence_ids(report)
                report_evidence_id = report_evidence_ids[0] if report_evidence_ids else None
                report_payload = {"summary": report.summary}
                if report_evidence_ids:
                    report_payload["evidence_ids"] = report_evidence_ids

                if self._unknown_paths:
                    self._publish(EventType.UNKNOWN_CHANGE_DETECTED, payload={"paths": sorted(self._unknown_paths)})
                if report.decision == Decision.PASS:
                    self._publish(
                        EventType.VERIFICATION_PASSED,
                        evidence_id=report_evidence_id,
                        payload=report_payload,
                    )
                elif report.decision == Decision.ROLLBACK:
                    self._publish(
                        EventType.VERIFICATION_FAILED,
                        evidence_id=report_evidence_id,
                        payload={"decision": "ROLLBACK", **report_payload},
                    )
                    self._publish(EventType.ROLLBACK_STARTED, evidence_id=report_evidence_id, payload=report_payload)
                    self._publish(EventType.ROLLBACK_COMPLETED, evidence_id=report_evidence_id, payload=report_payload)
                    self._publish(
                        EventType.TASK_FAILED,
                        evidence_id=report_evidence_id,
                        payload={"state": self.fsm.state.name, "error": f"Rolled back: {report.summary}", **report_payload},
                    )
                elif report.decision == Decision.BLOCKED:
                    self._publish(
                        EventType.VERIFICATION_FAILED,
                        evidence_id=report_evidence_id,
                        payload={"decision": "BLOCKED", **report_payload},
                    )
                    self._publish(
                        EventType.TASK_FAILED,
                        evidence_id=report_evidence_id,
                        payload={"state": self.fsm.state.name, "error": f"Blocked: {report.summary}", **report_payload},
                    )
                elif report.decision == Decision.TIMEOUT:
                    self._publish(
                        EventType.VERIFICATION_FAILED,
                        evidence_id=report_evidence_id,
                        payload={"decision": "TIMEOUT", **report_payload},
                    )
                    self._publish(EventType.TIMEOUT, evidence_id=report_evidence_id, payload=report_payload)
                    self._publish(
                        EventType.TASK_FAILED,
                        evidence_id=report_evidence_id,
                        payload={"state": self.fsm.state.name, "error": f"Timeout: {report.summary}", **report_payload},
                    )
                else:  # REPAIR
                    self._publish(
                        EventType.VERIFICATION_FAILED,
                        evidence_id=report_evidence_id,
                        payload={"decision": "REPAIR", **report_payload},
                    )

                if report.decision == Decision.PASS:
                    self.evidence.save()
                    self.fsm.complete(True)
                    self._publish(
                        EventType.TASK_COMPLETED,
                        evidence_id=report_evidence_id,
                        payload={"summary": last_summary, "state": self.fsm.state.name, **({"evidence_ids": report_evidence_ids} if report_evidence_ids else {})},
                    )
                    return AgentResult(
                        ok=True,
                        state=self.fsm.state.name,
                        summary=last_summary,
                        changes=changes,
                        verification=verification,
                        evidence=[item.to_dict() for item in self.evidence.get_all()],
                    )

                if report.decision == Decision.BLOCKED:
                    print("\nتم حجب المهمة بسبب تغييرات خارجية أو دليل مفقود.")
                    self.evidence.save()
                    if self.fsm.can_transition(State.FAILED):
                        self.fsm.transition(State.FAILED)
                    elif self.fsm.can_transition(State.REJECTED):
                        self.fsm.transition(State.REJECTED)
                    return AgentResult(
                        ok=False,
                        state=self.fsm.state.name,
                        summary=last_summary,
                        changes=changes,
                        verification=verification,
                        evidence=[item.to_dict() for item in self.evidence.get_all()],
                        error=f"Blocked: {report.summary}",
                    )

                if report.decision == Decision.ROLLBACK:
                    print("\nفشل غير قابل للإصلاح؛ سينتقل إلى ROLLED_BACK.")
                    self.evidence.save()
                    if self.fsm.can_transition(State.ROLLED_BACK):
                        self.fsm.transition(State.ROLLED_BACK)
                    elif self.fsm.can_transition(State.REJECTED):
                        self.fsm.transition(State.REJECTED)
                    return AgentResult(
                        ok=False,
                        state=self.fsm.state.name,
                        summary=last_summary,
                        changes=changes,
                        verification=verification,
                        evidence=[item.to_dict() for item in self.evidence.get_all()],
                        error=f"Rolled back: {report.summary}",
                    )

                if report.decision == Decision.TIMEOUT:
                    print("\nانتهت مهلة الخطوة.")
                    self.evidence.save()
                    if self.fsm.can_transition(State.FAILED):
                        self.fsm.transition(State.FAILED)
                    elif self.fsm.can_transition(State.REJECTED):
                        self.fsm.transition(State.REJECTED)
                    return AgentResult(
                        ok=False,
                        state=self.fsm.state.name,
                        summary=last_summary,
                        changes=changes,
                        verification=verification,
                        evidence=[item.to_dict() for item in self.evidence.get_all()],
                        error=f"Timeout: {report.summary}",
                    )

                # REPAIR: record this attempt's failure signature and continue.
                # The gate is the sole authority for breaking repeated-identical
                # failure loops: it returns ROLLBACK when the current signature
                # matches a prior attempt, so after two consecutive identical
                # failures the agent never attempts a third repair.
                self._repair_count += 1
                self._failure_signatures.append(
                    FailureSignature.from_report(report, self._changed_files)
                )

                # Transition to REPAIRING then back to EXECUTING
                if self.fsm.can_transition(State.REPAIRING):
                    self.fsm.transition(State.REPAIRING)
                print(f"\nتعذر اجتياز التحقق (محاولة إصلاح {self._repair_count})")
                failures = [result for result in verification if not result.ok]
                self.fsm.transition(State.EXECUTING)

            # Max rounds exhausted
            self.evidence.save()
            if self.fsm.can_transition(State.REJECTED):
                self.fsm.transition(State.REJECTED)
            self._publish(EventType.TASK_FAILED, payload={"summary": last_summary, "state": self.fsm.state.name, "error": "تم الوصول إلى الحد الأقصى لمحاولات الإصلاح"})
            return AgentResult(
                ok=False,
                state=self.fsm.state.name,
                summary=last_summary,
                changes=changes,
                verification=failures,
                evidence=[item.to_dict() for item in self.evidence.get_all()],
                error="تم الوصول إلى الحد الأقصى لمحاولات الإصلاح",
            )
        except (LLMError, FSMError, OSError, ValueError) as exc:
            if not self.fsm.is_terminal() and self.fsm.can_transition(State.REJECTED):
                self.fsm.transition(State.REJECTED)
            self._publish(EventType.TASK_FAILED, payload={"summary": "فشل تشغيل الوكيل", "state": self.fsm.state.name, "error": str(exc)})
            return AgentResult(False, self.fsm.state.name, "فشل تشغيل الوكيل", error=str(exc))
