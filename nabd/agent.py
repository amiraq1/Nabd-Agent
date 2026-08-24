"""Nabd coding-agent orchestration loop."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .evidence import EvidenceStore
from .fsm import FSM, FSMError, State
from .llm import LLMClient, LLMError
from .models import AgentResult, Plan, ToolCall, ToolResult
from .tools import ToolExecutor, format_call

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
    """Strip hallucinated tool-name prefixes from the verification list.

    The model occasionally echoes an action tool name at the start of a
    verification entry -- with or without a separator, e.g.
    ``"run_command: python hello.py"`` or ``"run_command python hello.py"`` --
    or emits a bare tool name like ``"run_command"``. Those are agent tools,
    not shell commands; executing them literally (``run_command: not found``)
    breaks the verification gate and wrongly REJECTS a successful task.
    Recover the real command by dropping a leading tool-name token (and any
    following ``:`` / whitespace) and filtering out bare tool names.
    """
    if not isinstance(commands, list):
        return []
    # Longest names first so "list_files" wins over "list" during matching.
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

    def run(self, task: str, max_rounds: int = 5) -> AgentResult:
        try:
            self.intent = "MUTATING" if self.workspace_free else classify_intent(task)
            self.executor.set_intent(self.intent)
            inventory = self.executor.execute(ToolCall("list_files", {"path": "."}))
            self.history.append(inventory)
            files = inventory.output
            failures: List[ToolResult] = []
            last_summary = ""
            changes: List[str] = []

            for round_number in range(1, max_rounds + 1):
                print(f"\n===== دورة الوكيل {round_number}/{max_rounds} =====")
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
                changes.extend(result.output for result in action_results if result.ok and result.name == "write_file")
                policy_failures = [
                    result
                    for result in action_results
                    if result.raw_facts is not None
                    and result.raw_facts.status == "MUTATION_NOT_ALLOWED"
                ]
                self.fsm.transition(State.VERIFYING)
                if policy_failures:
                    # A blocked mutation is not a verification failure that a
                    # shell command can override. Re-plan instead of allowing
                    # unrelated successful verification to prove completion.
                    failures = policy_failures
                    print("\nتم حجب تغيير خارج نطاق المهمة؛ سيعيد الوكيل التخطيط.")
                    self.fsm.transition(State.EXECUTING)
                    continue

                verification_calls = [ToolCall("run_command", {"command": command}) for command in plan.verification]
                verification = self._run_calls(verification_calls)
                for command, result in zip(plan.verification, verification):
                    if result.raw_facts is not None:
                        self.evidence.verify(
                            result.raw_facts,
                            claim=f"verification: {command}",
                            task_id=self.task_id,
                            relevant=result.ok,
                        )
                failures = [result for result in verification if not result.ok]
                if not failures and verification and self.evidence.is_usable_for_completion(self.task_id):
                    self.evidence.save()
                    self.fsm.complete(self.evidence.is_usable_for_completion(self.task_id))
                    return AgentResult(
                        ok=True,
                        state=self.fsm.state.name,
                        summary=last_summary,
                        changes=changes,
                        verification=verification,
                        evidence=[item.to_dict() for item in self.evidence.get_all()],
                    )

                print("\nتعذر اجتياز التحقق؛ سيحاول الوكيل إصلاح الأخطاء.")
                self.fsm.transition(State.EXECUTING)

            self.evidence.save()
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
