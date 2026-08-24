"""Workspace isolation and command safety checks for Nabd."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable


BLOCKED_PATH_PATTERNS = (
    re.compile(r"(?:^|/)\.git(?:/|$)", re.IGNORECASE),
    re.compile(r"(?:^|/)\.env(?:\.|/|$)", re.IGNORECASE),
    re.compile(r"(?:^|/)\.nabd(?:/|$)", re.IGNORECASE),
    re.compile(r"(?:^|/)\.ssh(?:/|$)", re.IGNORECASE),
    re.compile(r"(?:^|/)\.aws(?:/|$)", re.IGNORECASE),
    re.compile(r"(?:^|/)\.config(?:/|$)", re.IGNORECASE),
    re.compile(r"(?:/|^)__pycache__(?:/|$)", re.IGNORECASE),
    re.compile(r"(?:^|/)\.\.(?:/|$)"),
    re.compile(r"^/etc(?:/|$)"),
    re.compile(r"^/proc(?:/|$)"),
    re.compile(r"^/sys(?:/|$)"),
)

BLOCKED_COMMAND_PATTERNS = (
    re.compile(r"\brm\s+-[a-z]*r[a-z]*f[a-z]*\s+(?:/|~|\*)", re.IGNORECASE),
    re.compile(r"\brm\s+-[a-z]*f[a-z]*r[a-z]*\s+(?:/|~|\*)", re.IGNORECASE),
    re.compile(r"(?:curl|wget)[^\n|]*\s*\|\s*/?(?:(?:[A-Za-z0-9_.-]+/)*(?:ba)?sh)\b", re.IGNORECASE),
    re.compile(r"\beval\b", re.IGNORECASE),
    re.compile(r"\bexec\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|])\s*sudo\b", re.IGNORECASE),
    re.compile(r"\bchmod\s+(?:[0-7]*7[0-7]*|\+?r?wx)\b", re.IGNORECASE),
    re.compile(r">\s*/(?:etc|proc|sys)(?:\s|$)", re.IGNORECASE),
    re.compile(r"(?:^|[;&|])\s*dd\s+[^\n]*\bof=/dev/", re.IGNORECASE),
    re.compile(r"(?:^|[\s;&|])\.\.(?:[/\\]|$)", re.IGNORECASE),
)


class JailError(RuntimeError):
    """Raised when a workspace or command safety violation is detected."""


# Patterns blocked only for public (model-facing) paths.
# .nabd is included here so ReadTool/ListTool/SearchTool never expose
# internal artifacts to the model context.
PUBLIC_BLOCKED_PATH_PATTERNS = (
    *BLOCKED_PATH_PATTERNS,
)

# Patterns blocked for internal (trusted-code) paths.
# .nabd is *excluded* so WriteTool, verifier, and EvidenceStore can
# access .nabd/backups and .nabd/evidence.json.  All other blocks
# (root dirs, system paths, dotfiles) still apply.
INTERNAL_BLOCKED_PATH_PATTERNS = tuple(
    p for p in BLOCKED_PATH_PATTERNS
    if p.pattern != r"(?:^|/)\.nabd(?:/|$)"
)


class WorkspaceJail:
    """Enforces workspace boundaries and rejects known destructive patterns."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        if not self.workspace_root.is_dir():
            raise JailError(f"Workspace root does not exist: {workspace_root}")

    def _check_path_against(
        self, path: str | Path, allow_missing: bool, patterns: tuple
    ) -> Path:
        """Shared implementation for check_path / check_internal_path."""
        raw_path = str(path)
        if re.search(r"(?:^|[/\\])\.\.(?:[/\\]|$)", raw_path):
            raise JailError(f"Blocked traversal path: {path}")
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(self.workspace_root)
        except ValueError as exc:
            raise JailError(f"Path outside workspace: {path}") from exc

        path_string = str(candidate)
        for pattern in patterns:
            if pattern.search(path_string):
                raise JailError(f"Blocked path pattern: {path}")
        if not allow_missing and not candidate.exists():
            raise JailError(f"Path does not exist: {path}")
        return candidate

    def check_path(self, path: str | Path, allow_missing: bool = True) -> Path:
        """Public path check — blocks .nabd and all sensitive directories."""
        return self._check_path_against(path, allow_missing, PUBLIC_BLOCKED_PATH_PATTERNS)

    def check_internal_path(self, path: str | Path, allow_missing: bool = True) -> Path:
        """Internal path check — allows .nabd for trusted artifact access.

        Callers: WriteTool (backups), verifier (backups), EvidenceStore (save).
        This must NOT be reachable from ToolCall arguments or model data.
        """
        return self._check_path_against(path, allow_missing, INTERNAL_BLOCKED_PATH_PATTERNS)

    def check_command(self, command: str) -> None:
        if not command.strip():
            raise JailError("Empty shell command")
        for pattern in BLOCKED_COMMAND_PATTERNS:
            if pattern.search(command):
                raise JailError(f"Blocked command pattern: {command}")

    def safe_write(self, path: str | Path, content: str) -> str:
        target = self.check_path(path, allow_missing=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def safe_read(self, path: str | Path) -> str:
        target = self.check_path(path, allow_missing=False)
        if not target.is_file():
            raise JailError(f"Not a file: {path}")
        return target.read_text(encoding="utf-8")

    def is_safe_path(self, path: str | Path) -> bool:
        try:
            self.check_path(path)
            return True
        except JailError:
            return False

    def is_safe_command(self, command: str) -> bool:
        try:
            self.check_command(command)
            return True
        except JailError:
            return False
