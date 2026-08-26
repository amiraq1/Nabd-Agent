"""Tool-call adapter: tools return RawFacts; EvidenceStore verifies them later."""

from __future__ import annotations

from pathlib import Path
import shlex
import shutil
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
from .mutation import MutationController


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


# Shell builtins/keywords that may legitimately lead a command line passed to
# /bin/sh -c. These are structural markers: a natural-language planning sentence
# (human/LLM prose) does not begin with one of them, regardless of language.
_SHELL_BUILTINS = frozenset(
    {
        "cd", "echo", "export", "set", "unset", "source", ".", "read",
        "printf", "test", "[", "]", "if", "then", "else", "elif", "fi",
        "for", "while", "do", "done", "case", "esac", "function", "return",
        "exit", "break", "continue", "local", "let", "eval", "exec", "mapfile",
        "declare", "typeset", "alias", "unalias", "shift", "wait", "trap",
        "select", "until", "bg", "fg", "disown", "jobs",
    }
)

# Launcher prefixes that precede the real command (e.g. `sudo rm`, `env PY=1 cmd`).
_SHELL_LAUNCHERS = frozenset(
    {"sudo", "env", "time", "nohup", "stdbuf", "nice", "command", "bash", "sh", "zsh", "ksh", "csh"}
)


def is_plausible_shell_command(command: str) -> bool:
    """Return True only when *command* is shaped like a real shell command.

    The agent must never promote free natural-language text (a planning
    sentence or thought) into an executable shell command. A genuine command
    line begins with an executable resolvable on PATH, a shell builtin/keyword,
    or a known safe interpreter; prose does not. The check is structural and
    language-agnostic -- it never inspects the human language of the text, so
    Arabic, English, or any other prose is rejected for the same reason.
    """
    if not isinstance(command, str) or not command.strip():
        return False
    try:
        tokens = shlex.split(command.strip(), comments=False, posix=True)
    except ValueError:
        return False
    if not tokens:
        return False

    idx = 0
    # Skip launcher prefixes (sudo, env, sh -c, ...).
    while idx < len(tokens) and tokens[idx] in _SHELL_LAUNCHERS:
        idx += 1
    # Skip leading `NAME=VALUE` environment assignments.
    while idx < len(tokens) and "=" in tokens[idx] and not any(
        meta in tokens[idx] for meta in (";", "|", "&", ">", "<", "(", "$")
    ):
        idx += 1
    if idx >= len(tokens):
        return False

    head = tokens[idx]
    # Normalize a leading command group / subshell marker: (cd x && y) / { cd x; }.
    # shlex keeps the marker attached to the next word, so strip it off the head.
    if head and head[0] in "({":
        head = head[1:]

    if head in _SHELL_BUILTINS:
        return True
    if head in _READ_ONLY_SHELL_COMMANDS:
        return True
    # A flag to an interpreter/launcher (e.g. `bash -c`, `sh -c`) is itself a
    # deliberate command invocation; the wrapped program lives in the argument.
    if head.startswith("-"):
        return True
    candidate = Path(head).name
    if shutil.which(candidate) or shutil.which(head):
        return True
    return False


class ToolExecutor:
    def __init__(
        self,
        root: Path,
        approve: Optional[Callable[[ToolCall], bool]] = None,
        auto_approve: bool = False,
        command_timeout: int = 120,
        evidence_store: Optional[EvidenceStore] = None,
        controlled_mutation: bool = False,
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
        # M1 Controlled Mutation: when enabled, mutating operations must pass
        # the kernel-marker contract (verify_before / verify_after). Disabled
        # by default so existing behaviour and tests are unaffected.
        self.controlled = MutationController(self.root) if controlled_mutation else None

    def set_intent(self, intent: str) -> None:
        if intent not in {"READ_ONLY", "MUTATING"}:
            raise ValueError(f"Unknown task intent: {intent}")
        self.intent = intent

    def execute(self, call: ToolCall) -> ToolResult:
        try:
            allowed = {"list_files", "read_file", "write_file", "search", "run_command"}
            if call.name not in allowed:
                raise ToolError(f"Unknown tool: {call.name}")
            if call.name == "run_command":
                command = str(call.arguments.get("command", ""))
                if not is_plausible_shell_command(command):
                    return self._not_a_command(call, command)
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
            # M1 Controlled Mutation: route mutating operations through the
            # kernel-marker contract. Read-only commands and read-only tasks
            # are never gated here (they do not mutate).
            if self.controlled is not None:
                if call.name == "write_file":
                    raw = self.controlled.controlled_write(
                        self.writer,
                        str(call.arguments["path"]),
                        str(call.arguments["content"]),
                    )
                    return self._result(call.name, raw)
                if call.name == "run_command" and not is_read_only_shell_command(command):
                    raw = self.controlled.controlled_command(self.shell, command)
                    return self._result(call.name, raw)
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

    def _not_a_command(self, call: ToolCall, command: str) -> ToolResult:
        """Reject a run_command whose payload is not a real shell command.

        This is the structural firewall that stops natural-language planning
        text from ever reaching /bin/sh: only command-shaped input proceeds.
        """
        message = (
            "NOT_A_COMMAND: input was not a shell command; "
            "refusing to execute natural-language text"
        )
        raw = RawFacts(
            operation="shell",
            path=None,
            status="NOT_A_COMMAND",
            exit_code=126,
            error=message,
            details={
                "policy": "command-shape",
                "intent": self.intent,
                "tool": call.name,
                "command": command,
                "reason": "payload is not a recognized shell command",
            },
        )
        return ToolResult(call.name, False, message, 126, raw)

    @staticmethod
    def _mutation_denied(call: ToolCall) -> ToolResult:
        is_shell = call.name == "run_command"
        message = f"MUTATION_NOT_ALLOWED: task is READ_ONLY; {call.name} was blocked"
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
