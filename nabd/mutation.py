"""Controlled Mutation: gated filesystem / shell mutation behind verify_before / verify_after.

M1 introduces a *kernel marker* contract. The agent may only perform mutating
actions when the kernel marker is present and valid (``verify_before``), and
every mutation must leave the marker intact (``verify_after``). A failure of
``verify_before`` refuses the mutation outright; a failure of ``verify_after``
flags the mutation as a contract violation (rollback is handled by M2).

The kernel marker lives under ``.nabd/KERNEL_MARKER``. Because ``.nabd`` is a
protected (internal-only) path for public tooling, the marker cannot be
clobbered through the normal ``write_file`` API; only trusted internal code
(``ensure_marker``) or a shell command may touch it. That asymmetry is exactly
what makes it a meaningful integrity sentinel.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from .raw_facts import RawFacts

KERNEL_MARKER_RELPATH = ".nabd/KERNEL_MARKER"
KERNEL_MARKER_CONTENT = "Controlled mutation marker: Nabd M1 smoke."

CONTROLLED_MUTATION_BLOCKED = "CONTROLLED_MUTATION_BLOCKED"
CONTROLLED_MUTATION_VERIFY_FAILED = "CONTROLLED_MUTATION_VERIFY_FAILED"


class MutationController:
    """Gate mutating operations on the kernel-marker contract."""

    def __init__(
        self,
        root: Path,
        marker_relpath: str = KERNEL_MARKER_RELPATH,
        marker_content: str = KERNEL_MARKER_CONTENT,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.marker_rel = Path(marker_relpath)
        self.marker_path = self.root / self.marker_rel
        self.marker_content = marker_content

    # ------------------------------------------------------------------
    # Marker lifecycle (trusted internal code; bypasses the public jail).
    # ------------------------------------------------------------------
    def ensure_marker(self) -> bool:
        """Idempotently create the kernel marker. Returns True if now valid."""
        try:
            self.marker_path.parent.mkdir(parents=True, exist_ok=True)
            if self.marker_valid():
                return True
            self.marker_path.write_text(self.marker_content, encoding="utf-8")
            return self.marker_valid()
        except OSError:
            return False

    def remove_marker(self) -> None:
        """Remove the kernel marker (simulate a missing marker)."""
        try:
            self.marker_path.unlink()
        except FileNotFoundError:
            pass

    def corrupt_marker(self, content: str = "CORRUPTED") -> None:
        """Overwrite the marker with invalid content (simulate corruption)."""
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        self.marker_path.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Marker inspection.
    # ------------------------------------------------------------------
    def _read_marker(self) -> Optional[str]:
        try:
            return self.marker_path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return None

    def marker_present(self) -> bool:
        return self._read_marker() is not None

    def marker_valid(self) -> bool:
        return self._read_marker() == self.marker_content

    # ------------------------------------------------------------------
    # Contract predicates.
    # ------------------------------------------------------------------
    def verify_before(self) -> bool:
        """Safe to mutate only if the kernel marker is present and valid."""
        return self.marker_valid()

    def verify_after(self) -> bool:
        """Mutation conforms only if the kernel marker remains intact."""
        return self.marker_valid()

    # ------------------------------------------------------------------
    # Target snapshot (pre-mutation hash, for evidence).
    # ------------------------------------------------------------------
    def snapshot_target(self, path: str) -> Optional[str]:
        """SHA-256 of the mutation target before the write (None if absent)."""
        try:
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = self.root / candidate
            candidate = candidate.resolve()
            candidate.relative_to(self.root)
        except (ValueError, OSError):
            return None
        if not candidate.is_file():
            return None
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    # ------------------------------------------------------------------
    # Gated mutation wrappers.
    # ------------------------------------------------------------------
    def controlled_write(self, writer, path: str, content: str) -> RawFacts:
        before_ok = self.verify_before()
        if not before_ok:
            return self._refused(
                "write",
                path,
                "verify_before failed: kernel marker missing or invalid",
            )
        before_sha = self.snapshot_target(path)
        raw = writer.run(path, content)
        after_ok = self.verify_after()
        contract = self._contract(before_ok, after_ok, before_sha)
        if not after_ok:
            return dataclasses.replace(
                raw,
                status=CONTROLLED_MUTATION_VERIFY_FAILED,
                details={**raw.details, "controlled_mutation": contract},
            )
        return dataclasses.replace(
            raw,
            details={**raw.details, "controlled_mutation": contract},
        )

    def controlled_command(self, shell_tool, command: str) -> RawFacts:
        before_ok = self.verify_before()
        if not before_ok:
            return self._refused(
                "shell",
                None,
                "verify_before failed: kernel marker missing or invalid",
                command=command,
            )
        raw = shell_tool.run(command)
        after_ok = self.verify_after()
        contract = self._contract(before_ok, after_ok, None)
        if not after_ok:
            return dataclasses.replace(
                raw,
                status=CONTROLLED_MUTATION_VERIFY_FAILED,
                details={**raw.details, "controlled_mutation": contract},
            )
        return dataclasses.replace(
            raw,
            details={**raw.details, "controlled_mutation": contract},
        )

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------
    def _contract(
        self, before_ok: bool, after_ok: bool, before_sha: Optional[str]
    ) -> Dict[str, Any]:
        return {
            "verify_before": before_ok,
            "verify_after": after_ok,
            "marker_valid": self.marker_valid(),
            "marker_present": self.marker_present(),
            "before_sha256": before_sha,
        }

    def _refused(
        self,
        operation: str,
        path: Optional[str],
        reason: str,
        command: Optional[str] = None,
    ) -> RawFacts:
        return RawFacts(
            operation=operation,
            path=str(path) if path is not None else None,
            status=CONTROLLED_MUTATION_BLOCKED,
            exit_code=126,
            error=reason,
            stdout="",
            stderr=reason,
            details={
                "policy": "controlled_mutation",
                "marker_present": self.marker_present(),
                "marker_valid": self.marker_valid(),
                "command": command,
                "reason": reason,
            },
        )
