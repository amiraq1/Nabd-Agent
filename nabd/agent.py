"""Nabd coding-agent orchestration loop."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .evidence import EvidenceStore
from .fsm import FSM, FSMError, State
from .llm import LLMClient, LLMError
from .models import AgentResult, Plan, ToolCall, ToolResult
from .tools import ToolExecutor, format_call
from .verify.gate import VerificationGate, diff_snapshots, take_snapshot
from .verify.types import (
    Criterion,
    CriterionKind,
    Decision,
    FailureSignature,
    Report,
    SuccessCriteria,
)

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


def classify_intent(task: str) -> str:
    """Classify task scope before planning; unknown tasks remain mutating."""
    text = " ".join(str(task).lower().split())
    if any(keyword in text for keyword in MUTATING_KEYWORDS):
        return "MUTATING"
    if any(keyword in text for keyword in READ_ONLY_KEYWORDS):
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
    ) -> None:
        self.client = LLMClient(provider)
        self.fsm = FSM()
        self.history: List[ToolResult] = []
        self.auto_approve = auto_approve or workspace_free
        self.workspace_free = workspace_free
        self.intent = "MUTATING"
        self.task_id = EvidenceStore.new_task_id()
        self.evidence = EvidenceStore(root, task_id=self.task_id)
        self.executor = ToolExecutor(
            root,
            approve=self._approve,
            auto_approve=self.auto_approve,
            evidence_store=self.evidence,
        )
        # Verification Gate state
        self._snapshot_before: Dict[str, str] = {}
        self._snapshot_after: Dict[str, str] = {}
        self._unknown_paths: Set[str] = set()
        self._changed_files: List[str] = []
        self._repair_count: int = 0
        self._budget_spent: int = 0
        self._last_failure_signature: Optional[FailureSignature] = None
        self._start_time: float = 0.0

    def _approve(self, call: ToolCall) -> bool:
        print(f"\nطلب الوكيل تنفيذ: {format_call(call)}")
        answer = input("السماح؟ [y/N]: ").strip().lower()
        return answer in {"y", "yes", "نعم", "ن"}

    def _run_calls(self, calls: List[ToolCall]) -> List[ToolResult]:
        results: List[ToolResult] = []
        for call in calls:
            print(f"\n[أداة] {format_call(call)}")
            result = self.executor.execute(call)
            self.history.append(result)
            results.append(result)
            marker = "نجاح" if result.ok else "فشل"
            print(f"[{marker}] {result.output[:4000]}")
        return results

    def _ask(self, task: str, files: str, failures: List[ToolResult]) -> Plan:
        response = self.client.complete_json(
            SYSTEM_PROMPT,
            _context(task, files, self.history, failures),
        )
        return _as_plan(response)

    def _take_snapshot_before(self) -> None:
        """Take a filesystem snapshot before task execution begins."""
        try:
            self._snapshot_before = take_snapshot(self.executor.root)
        except Exception:
            self._snapshot_before = {}

    def _take_snapshot_after(self) -> None:
        """Take a filesystem snapshot after execution and detect unknown changes."""
        try:
            self._snapshot_after = take_snapshot(self.executor.root)
            diff = diff_snapshots(self._snapshot_before, self._snapshot_after)
            # Unknown paths = changes not in our changed_files list
            all_changed = set(diff["added"] + diff["removed"] + diff["changed"])
            known = set(self._changed_files)
            self._unknown_paths = all_changed - known
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
                )

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
            self.intent = "MUTATING" if self.workspace_free else classify_intent(task)
            self.executor.set_intent(self.intent)
            inventory = self.executor.execute(ToolCall("list_files", {"path": "."}))
            self.history.append(inventory)
            files = inventory.output
            failures: List[ToolResult] = []
            last_summary = ""
            changes: List[str] = []

            # Take initial snapshot for unknown-change detection
            self._take_snapshot_before()

            for round_number in range(1, max_rounds + 1):
                print(f"\n===== دورة الوكيل {round_number}/{max_rounds} =====")

                # Check wall-clock timeout
                if self._check_timeout():
                    print("\nانتهت مهلة المهمة؛ سينتقل إلى FAILED.")
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
                        verification=failures,
                        evidence=[item.to_dict() for item in self.evidence.get_all()],
                        error="Timeout: wall-clock budget exceeded",
                    )

                if self.fsm.state == State.PLANNING:
                    plan = self._ask(task, files, failures)
                    last_summary = plan.summary
                    print(f"\nالخطة: {plan.summary}")
                    for index, step in enumerate(plan.steps, 1):
                        print(f"  {index}. {step}")
                    self.fsm.transition(State.EXECUTING)
                else:
                    # After a failed verification the FSM is already EXECUTING;
                    # request a repair plan without a redundant self-transition.
                    plan = self._ask(task, files, failures)
                    last_summary = plan.summary

                action_results = self._run_calls(plan.tool_calls)
                for call, result in zip(plan.tool_calls, action_results):
                    if result.raw_facts is not None:
                        self.evidence.verify(
                            result.raw_facts,
                            claim=f"tool {call.name}",
                            task_id=self.task_id,
                            relevant=call.name in {"write_file", "read_file", "search", "run_command"},
                        )
                # Track changed files
                for result in action_results:
                    if result.ok and result.name == "write_file":
                        # Extract path from the tool call
                        for call in plan.tool_calls:
                            if call.name == "write_file" and call.arguments.get("path"):
                                self._changed_files.append(str(call.arguments["path"]))
                                break
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

                verification_calls = [ToolCall("run_command", {"command": command}) for command in plan.verification]
                verification = self._run_calls(verification_calls)
                self._record_verification_evidence(verification, plan)

                # Use the Verification Gate for decision
                report = self._run_with_gate(task, plan, action_results, verification)

                print(f"\n[بوابة التحقق] القرار: {report.decision.value}")
                print(f"  {report.summary}")

                if report.decision == Decision.PASS:
                    self.evidence.save()
                    self.fsm.complete(True)
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

                # REPAIR: track failure signature and repair count
                self._repair_count += 1
                current_sig = FailureSignature.from_report(report, self._changed_files)
                if (
                    self._last_failure_signature is not None
                    and current_sig.signature == self._last_failure_signature.signature
                    and current_sig.file_set == self._last_failure_signature.file_set
                ):
                    current_sig.no_improvement_streak = (
                        self._last_failure_signature.no_improvement_streak + 1
                    )
                self._last_failure_signature = current_sig

                # Check no-improvement streak (2+ identical failures -> ROLLBACK)
                if current_sig.no_improvement_streak >= 2:
                    print("\nفشل متكرر بلا تحسن؛ سينتقل إلى ROLLED_BACK.")
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
                        error=f"Rolled back: repeated failure with no improvement",
                    )

                # Transition to REPAIRING then back to EXECUTING
                if self.fsm.can_transition(State.REPAIRING):
                    self.fsm.transition(State.REPAIRING)
                print(f"\nتعذر اجتياز التحقق (محاولة إصلاح {self._repair_count}/{3})")
                failures = [result for result in verification if not result.ok]
                self.fsm.transition(State.EXECUTING)

            # Max rounds exhausted
            self.evidence.save()
            if self.fsm.can_transition(State.REJECTED):
                self.fsm.transition(State.REJECTED)
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
            return AgentResult(False, self.fsm.state.name, "فشل تشغيل الوكيل", error=str(exc))
