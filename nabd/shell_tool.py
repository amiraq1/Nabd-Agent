"""Raw-facts, jail-checked shell execution with process-group timeout.

Commands execute WITHOUT a shell (``shell=False``) in their own process group
(``start_new_session=True``). On timeout the entire process group is terminated
(SIGTERM, then SIGKILL after a documented grace period) and reaped, so no
child or grandchild can survive. ``start_time`` is captured at spawn to guard
against PID/PGID reuse after the process exits.
"""

from __future__ import annotations

import os
import signal
import shlex
import subprocess
import time
from pathlib import Path

from .jail import WorkspaceJail
from .raw_facts import RawFacts

# Explicit, documented grace period before escalating SIGTERM -> SIGKILL.
GRACE_PERIOD_MS = 3000


def _read_process_start_time(pid: int) -> float | None:
    """Return process start_time (clock ticks since boot) or None if unavailable."""
    try:
        with open(f"/proc/{pid}/stat", "r") as fh:
            data = fh.read()
        left = data.index("(")
        right = data.rindex(")")
        rest = data[right + 1 :].split()
        # start_time is overall field 22; rest[0] == field 3, so index 19.
        return float(rest[19])
    except (OSError, ValueError, IndexError):
        return None


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
            argv = shlex.split(command, posix=True)
        except ValueError as exc:
            return RawFacts(
                operation="shell", status="TOOL_ERROR", error=f"cannot parse command: {exc}"
            )
        if not argv:
            return RawFacts(operation="shell", status="TOOL_ERROR", error="empty command")

        started_at = time.monotonic()
        try:
            proc = subprocess.Popen(
                argv,
                shell=False,
                cwd=str(self.root),
                start_new_session=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, ValueError) as exc:
            return RawFacts(operation="shell", status="TOOL_ERROR", error=str(exc))

        # Capture PGID and start_time ONCE, immediately after spawn.
        pgid = os.getpgid(proc.pid)
        pgid_start_time = _read_process_start_time(proc.pid)
        liveness_method = "PROC_FS" if pgid_start_time is not None else "KILL_SIGNAL_PROBE"

        try:
            stdout, stderr = proc.communicate(timeout=self.timeout)
            duration = int((time.monotonic() - started_at) * 1000)
            return RawFacts(
                operation="shell",
                exit_code=proc.returncode,
                stdout=(stdout or "")[:8000],
                stderr=(stderr or "")[:4000],
                signal=signal.Signals(-proc.returncode).value if proc.returncode < 0 else None,
                status="OK",
                details={
                    "command": command,
                    "outcome": "COMPLETED",
                    "timeout_ms": self.timeout * 1000,
                    "grace_period_ms": GRACE_PERIOD_MS,
                    "duration_ms": duration,
                    "termination_signal": None,
                    "pid": proc.pid,
                    "pgid": pgid,
                    "pgid_start_time": pgid_start_time,
                    "children_remaining": 0,
                    "liveness_check_method": liveness_method,
                    "process_group_isolated": pgid != os.getpgrp() and pgid > 2,
                },
            )
        except subprocess.TimeoutExpired:
            termination_signal = self._terminate_group(proc, pgid)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                # Process (or a surviving descendant) still holds the pipe;
                # never block on read. Abandon the pipe and reap what we can.
                try:
                    proc.stdout.close()
                    proc.stderr.close()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                stdout, stderr = "", ""
            except (OSError, ValueError):
                stdout, stderr = "", ""
            duration = int((time.monotonic() - started_at) * 1000)
            children_remaining = self._count_remaining(pgid)
            status = (
                "TIMEOUT_TERMINATION_FAILED"
                if termination_signal == "TERMINATION_FAILED"
                else "TIMEOUT"
            )
            return RawFacts(
                operation="shell",
                exit_code=124,
                stdout=(stdout or "")[:8000],
                stderr=(stderr or "")[:4000],
                status=status,
                error=(
                    "Command exceeded timeout; process group termination failed"
                    if status == "TIMEOUT_TERMINATION_FAILED"
                    else "Command exceeded timeout; process group terminated"
                ),
                details={
                    "command": command,
                    "outcome": "TIMEOUT",
                    "timeout_ms": self.timeout * 1000,
                    "grace_period_ms": GRACE_PERIOD_MS,
                    "duration_ms": duration,
                    "termination_signal": (
                        None if termination_signal == "TERMINATION_FAILED" else termination_signal
                    ),
                    "pid": proc.pid,
                    "pgid": pgid,
                    "pgid_start_time": pgid_start_time,
                    "children_remaining": children_remaining,
                    "liveness_check_method": liveness_method,
                    "process_group_isolated": pgid != os.getpgrp() and pgid > 2,
                },
            )

    def _terminate_group(self, proc, pgid: int) -> str:
        """Terminate the captured process group: SIGTERM, grace, SIGKILL, reap.

        Never touches the current session/group. Returns the signal that killed
        the group, or "TERMINATION_FAILED" if delivery was impossible.
        """
        if pgid == os.getpgrp() or pgid <= 2:
            return "TERMINATION_FAILED"
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            self._reap_group(pgid)
            return "SIGTERM"
        except (PermissionError, OSError):
            return "TERMINATION_FAILED"

        deadline = time.monotonic() + (GRACE_PERIOD_MS / 1000.0)
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.02)

        if proc.poll() is None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                return "TERMINATION_FAILED"
            time.sleep(0.1)
        self._reap_group(pgid)
        return "SIGKILL" if proc.poll() is not None else "TERMINATION_FAILED"

    def _reap_group(self, pgid: int) -> None:
        try:
            while True:
                pid, _ = os.waitpid(-pgid, os.WNOHANG)
                if pid == 0:
                    break
        except (ChildProcessError, OSError):
            pass

    def _count_remaining(self, pgid: int) -> int:
        try:
            remaining = 0
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    stat = (entry / "stat").read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                left = stat.find("(")
                right = stat.rfind(")")
                if left == -1 or right == -1 or right <= left:
                    continue
                rest = stat[right + 1 :].split()
                if len(rest) < 3:
                    continue
                try:
                    pgrp = int(rest[2])
                except ValueError:
                    continue
                if pgrp == pgid and rest[0] != "Z":
                    remaining += 1
        except OSError:
            return 0
        return remaining
