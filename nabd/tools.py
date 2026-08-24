"""Tool-call adapter: tools return RawFacts; EvidenceStore verifies them later."""

from __future__ import annotations

from pathlib import Path
import shlex
from typing import Callable, Optional

from .models import ToolCall, ToolResult
from .jail import JailError
from .raw_facts import RawFacts
from .evidence import EvidenceStore
from .list_tool import ListTool
from .read_tool import ReadTool
from .search_tool import SearchTool
from .shell_tool import ShellTool
from .write_tool import WriteTool


class ToolError(RuntimeError):
    """Raised for invalid tool requests or approval rejection."""


_READ_ONLY_SHELL_COMMANDS = frozenset(
    {
        "cat",
        "command",
        "file",
        "find",
        "grep",
        "head",
        "ls",
        "pwd",
        "rg",
        "sha256sum",
        "stat",
        "tail",
        "type",
        "which",
        "wc",
    }
)
_READ_ONLY_GIT_COMMANDS = frozenset(
    {"branch", "diff", "log", "ls-files", "rev-parse", "show", "status"}
)
_FIND_MUTATING_OPTIONS = frozenset(
    {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprintf", "-fls"}
)


def is_read_only_shell_command(command: str) -> bool:
    """Return true only for a conservative, operator-free inspection command."""
    if not isinstance(command, str) or not command.strip():
        return False
    if any(operator in command for operator in (";", "|", "&", ">", "`", "$(", "\\")):
        return False
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False
    executable = Path(tokens[0]).name
    if executable == "git":
        if len(tokens) < 2 or tokens[1] not in _READ_ONLY_GIT_COMMANDS:
            return False
        if tokens[1] == "status":
            return "--no-optional-locks" in tokens[2:]
        return True
    if executable not in _READ_ONLY_SHELL_COMMANDS:
        return False
    if executable == "find" and any(token in _FIND_MUTATING_OPTIONS for token in tokens[1:]):
        return False
    return True


class ToolExecutor:
    def __init__(
        self,
        root: Path,
        approve: Optional[Callable[[ToolCall], bool]] = None,
        auto_approve: bool = False,
        command_timeout: int = 120,
        evidence_store: Optional[EvidenceStore] = None,
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.approve = approve
        self.auto_approve = auto_approve
        self.command_timeout = command_timeout
        self.evidence = evidence_store
        self.intent = "MUTATING"
        self.writer = WriteTool(self.root)
        self.reader = ReadTool(self.root)
        self.lister = ListTool(self.root)
        self.searcher = SearchTool(self.root)
        self.shell = ShellTool(self.root, timeout=command_timeout)

    def set_intent(self, intent: str) -> None:
        if intent not in {"READ_ONLY", "MUTATING"}:
            raise ValueError(f"Unknown task intent: {intent}")
        self.intent = intent

    def execute(self, call: ToolCall) -> ToolResult:
        try:
            allowed = {"list_files", "read_file", "write_file", "search", "run_command"}
            if call.name not in allowed:
                raise ToolError(f"Unknown tool: {call.name}")
            if self.intent == "READ_ONLY" and (
                call.name == "write_file"
                or (
                    call.name == "run_command"
                    and not is_read_only_shell_command(str(call.arguments.get("command", "")))
                )
            ):
                return self._mutation_denied(call)
            if call.name in {"write_file", "search", "run_command"} and not self._approved(call):
                return ToolResult(call.name, False, "Action rejected by user", 126)
            raw = self._dispatch(call)
            return self._result(call.name, raw)
        except (ToolError, JailError, OSError, ValueError) as exc:
            return ToolResult(call.name, False, str(exc), 1)

    def _approved(self, call: ToolCall) -> bool:
        if self.auto_approve:
            return True
        return bool(self.approve and self.approve(call))

    def _dispatch(self, call: ToolCall) -> RawFacts:
        args = call.arguments
        if call.name == "write_file":
            if "path" not in args or "content" not in args:
                raise ToolError("write_file requires path and content")
            return self.writer.run(str(args["path"]), str(args["content"]))
        if call.name == "read_file":
            return self.reader.run(str(args["path"]), int(args.get("max_bytes", 120_000)))
        if call.name == "list_files":
            return self.lister.run(str(args.get("path", ".")))
        if call.name == "search":
            return self.searcher.run(
                str(args.get("query", "")),
                str(args.get("path", ".")),
                int(args.get("max_results", 50)),
            )
        return self.shell.run(str(args.get("command", "")))

    @staticmethod
    def _mutation_denied(call: ToolCall) -> ToolResult:
        message = f"MUTATION_NOT_ALLOWED: task is READ_ONLY; {call.name} was blocked"
        is_shell = call.name == "run_command"
        raw = RawFacts(
            operation="shell" if is_shell else "write",
            path=None if is_shell else str(call.arguments.get("path", "")) or None,
            status="MUTATION_NOT_ALLOWED",
            exit_code=126,
            error=message,
            details={
                "policy": "mutation",
                "intent": "READ_ONLY",
                "tool": call.name,
                "command": str(call.arguments.get("command", "")) if is_shell else None,
                "reason": "read-only task cannot mutate workspace",
            },
        )
        return ToolResult(call.name, False, message, 126, raw)

    @staticmethod
    def _result(name: str, raw: RawFacts) -> ToolResult:
        if raw.operation == "write":
            output = f"Wrote {raw.path}" if raw.successful else (raw.error or "write failed")
        elif raw.operation == "read":
            output = raw.stdout if raw.successful else (raw.error or "read failed")
        elif raw.operation == "list":
            files = raw.details.get("files", [])
            output = "\n".join(files) or "(empty)"
            if raw.truncated:
                output += "\n...[listing truncated]"
        elif raw.operation == "search":
            matches = raw.details.get("matches", [])
            output = "\n".join(matches) or "(no matches)"
            if raw.details.get("fallback_used"):
                output += "\n[backend: python fallback]"
        else:
            output = raw.stdout or raw.stderr or raw.error or "(no output)"
        return ToolResult(name, raw.successful, output, raw.exit_code, raw)


def format_call(call: ToolCall) -> str:
    if call.name == "search":
        return f"search: {call.arguments.get('query', '')}"
    if call.name == "run_command":
        return f"run_command: {call.arguments.get('command', '')}"
    if call.name == "write_file":
        content = str(call.arguments.get("content", ""))
        return f"write_file: {call.arguments.get('path', '')} ({len(content)} bytes)"
    return f"{call.name}: {call.arguments}"
