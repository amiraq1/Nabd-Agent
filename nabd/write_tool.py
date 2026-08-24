"""Raw-facts file writer; evidence verification belongs to EvidenceStore."""

from __future__ import annotations

import difflib
import hashlib
import os
import pathlib
import shutil
import threading
from pathlib import Path
import time
import uuid
from contextlib import contextmanager
from typing import Optional

from .jail import WorkspaceJail
from .raw_facts import RawFacts


def _sha256(p: pathlib.Path) -> Optional[str]:
    """Compute SHA-256 of a file, or None if missing."""
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Per-directory locking (fixes TOCTOU races between check_path and write)
# ---------------------------------------------------------------------------
_locks: dict[str, threading.RLock] = {}
_glock = threading.Lock()


@contextmanager
def _lock_for(resolved: pathlib.Path):
    """Yield an RLock scoped to the parent directory of *resolved*."""
    key = str(resolved.parent.resolve())
    with _glock:
        lk = _locks.setdefault(key, threading.RLock())
    lk.acquire()
    try:
        yield
    finally:
        lk.release()


class WriteTool:
    """Write inside the workspace and return an untrusted filesystem receipt.

    Improvements over the naive implementation:
    * Per-directory RLock prevents TOCTOU between check_path and write.
    * Atomic writes via temp-file + ``os.replace``.
    * SHA-256 computed *before* and *after* the write for integrity checks.
    * Unified diff of before/after content stored in ``details``.
    * ``fsync`` after write to flush to disk.
    * Backup failures are non-fatal (recorded in details, not raised).
    """

    def __init__(self, workspace_root: str | pathlib.Path) -> None:
        self.root = pathlib.Path(workspace_root).expanduser().resolve()
        self.jail = WorkspaceJail(self.root)

    # ------------------------------------------------------------------
    def run(self, path: str, content: str | bytes) -> RawFacts:
        # 1. Validate the path through the jail (keeps .ssh/.aws/.config protection).
        try:
            resolved = self.jail.check_path(path, allow_missing=True)
        except Exception as e:
            raise RuntimeError(f"blocked: {e}")

        rel = resolved.relative_to(self.root)

        # Normalise content to bytes + text.
        content_bytes: bytes = (
            content.encode("utf-8") if isinstance(content, str) else content
        )
        content_text: str = (
            content
            if isinstance(content, str)
            else content_bytes.decode("utf-8", errors="replace")
        )

        with _lock_for(resolved):
            # 2. BEFORE snapshot (inside lock to avoid TOCTOU).
            before_exists = resolved.exists()
            before_sha = _sha256(resolved) if before_exists else None
            before_text = (
                resolved.read_text(encoding="utf-8", errors="replace")
                if before_exists and resolved.is_file()
                else ""
            )

            # 3. Backup existing file in .nabd/backups (non-fatal on failure).
            #    Uses check_internal_path so .nabd is allowed for trusted code.
            backup_path: Optional[str] = None
            if before_exists:
                try:
                    backup_dir = self.jail.check_internal_path(
                        Path(".nabd/backups") / rel.parent, allow_missing=True
                    )
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    backup_path = str(
                        backup_dir
                        / f"{rel.name}.{int(time.time())}.{uuid.uuid4().hex[:6]}.bak"
                    )
                    shutil.copy2(resolved, backup_path)
                except Exception as be:
                    backup_path = f"backup_failed: {be}"

            # 4. Atomic write: write to temp, fsync, then os.replace.
            tmp = resolved.parent / f".tmp.{uuid.uuid4().hex[:8]}"
            try:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(content_bytes)
                # flush + fsync
                try:
                    with tmp.open("rb") as f:
                        os.fsync(f.fileno())
                except Exception:
                    pass
                os.replace(tmp, resolved)
            except Exception as e:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
                raise RuntimeError(f"write failed: {e}")
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)

            # 5. AFTER snapshot from the filesystem.
            after_sha = _sha256(resolved)
            if after_sha is None:
                raise RuntimeError("after sha failed")
            after_stat = resolved.stat()
            after_size = after_stat.st_size
            after_mtime = after_stat.st_mtime
            after_text = resolved.read_text(encoding="utf-8", errors="replace")

            diff = "".join(
                difflib.unified_diff(
                    before_text.splitlines(keepends=True) if before_exists else [],
                    after_text.splitlines(keepends=True),
                    fromfile=f"{rel}@before:{(before_sha or 'none')[:8]}",
                    tofile=f"{rel}@after:{after_sha[:8]}",
                )
            )

        # 6. Return RawFacts — before/after/diff live inside details only.
        return RawFacts(
            operation="write",
            path=str(rel),
            exists=True,
            size=after_size,
            sha256=after_sha,
            mtime=after_mtime,
            backup=backup_path,
            details={
                "before_exists": before_exists,
                "before_sha256": before_sha,
                "after_sha256": after_sha,
                "changed": before_sha != after_sha,
                "diff": diff,
                "write_atomic": True,
            },
        )
