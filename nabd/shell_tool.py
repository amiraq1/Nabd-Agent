"""Raw-facts, jail-checked shell execution."""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path

from .jail import WorkspaceJail
from .raw_facts import RawFacts


class ShellTool:
    """Run a checked command and return execution facts only."""

    def __init__(self, workspace_root: str | Path, timeout: int = 120) -> None:
        self.root = Path(workspace_root).expanduser().resolve()
        self.jail = WorkspaceJail(self.root)
        self.timeout = max(1, min(int(timeout), 120))

    def run(self, command: str) -> RawFacts:
        try:
            self.jail.check_command(command)
        except Exception as exc:
            return RawFacts(operation="shell", status="TOOL_ERROR", error=str(exc))
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            return RawFacts(
                operation="shell",
                exit_code=result.returncode,
                stdout=(result.stdout or "")[:8000],
                stderr=(result.stderr or "")[:4000],
                signal=signal.Signals(-result.returncode).value if result.returncode < 0 else None,
                details={"command": command},
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return RawFacts(
                operation="shell",
                exit_code=124,
                truncated=True,
                stdout=stdout[:8000],
                stderr=(stderr + "\nCommand timed out")[:4000],
                details={"command": command, "timeout": True},
            )
