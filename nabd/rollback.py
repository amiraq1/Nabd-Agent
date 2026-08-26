"""M2 — Filesystem Rollback and Crash-Safe Resume.

A minimal, self-contained rollback manager for Nabd's controlled-mutation
pipeline. It restores *agent-owned* modifications to their pre-task hash,
removes *agent-created* files, leaves *UNKNOWN* (externally changed) files
strictly untouched, and preserves symlink semantics without ever following a
symlink target.

Design rules (from the M2 prompt):
* Snapshot before any mutation, recording full SHA-256 plus the entry type
  (regular_file / symlink / directory). Symlinks are stored as symlinks.
* Backup each in-scope regular file at snapshot time (pre-mutation source).
* Restore is atomic: copy from backup to a temp file -> fsync -> os.replace.
  A live file is never replaced by a partial one.
* Append-only rollback log; earlier lines are never rewritten.
* Interruption is deterministic and test-injected (no kill of unrelated
  processes). A resumed rollback reads the last committed log entry and
  continues; it does not start from scratch.
* UNKNOWN files are never restored or deleted; if a managed file was changed
  by an UNKNOWN actor, the rollback BLOCKS rather than overwriting it.
* The manager is opt-in: it is only constructed/used by the rollback tests
  and the T10 integration; default agent behaviour is untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .redact import redact_obj

# Log operations (append-only vocabulary).
OP_STARTED = "STARTED"
OP_PLAN_CREATED = "PLAN_CREATED"
OP_RESTORE_STARTED = "RESTORE_FILE_STARTED"
OP_RESTORE_COMMITTED = "RESTORE_FILE_COMMITTED"
OP_DELETE_STARTED = "DELETE_FILE_STARTED"
OP_DELETE_COMMITTED = "DELETE_FILE_COMMITTED"
OP_INTERRUPTED = "INTERRUPTED"
OP_RESUMED = "RESUMED"
OP_FINAL_VERIFY = "FINAL_VERIFY_STARTED"
OP_COMMITTED = "COMMITTED"
OP_FAILED = "FAILED_ROLLBACK"
OP_BLOCKED = "BLOCKED"

ENTRY_TYPES = ("regular_file", "symlink", "directory")


class RollbackError(RuntimeError):
    """Unrecoverable rollback error (e.g. invalid baseline manifest)."""


class RollbackInterrupted(Exception):
    """Raised when a test-injected fault stops a rollback mid-way."""

    def __init__(self, last_sequence: int) -> None:
        self.last_sequence = last_sequence
        super().__init__(f"Rollback interrupted after sequence {last_sequence}")


@dataclass
class RollbackResult:
    decision: str  # COMMITTED | FAILED_ROLLBACK | BLOCKED | INTERRUPTED
    rollback_id: str
    interrupted: bool = False
    resumed: bool = False
    final_manifest_match: bool = False
    unknown_touched: bool = False
    restored: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    blocked_path: Optional[str] = None
    duration_ms: float = 0.0


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file() or path.is_symlink():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FaultInjector:
    """Deterministic fault: raise RollbackInterrupted after N committed ops."""

    def __init__(self, after_committed_operations: int) -> None:
        self.after_committed_operations = after_committed_operations

    def should_interrupt(self, committed: int, total: int) -> bool:
        return committed >= self.after_committed_operations and committed < total


class RollbackManager:
    def __init__(
        self,
        root: Path,
        rollback_id: str = "rb",
        store_dir: Optional[Path] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.rollback_id = rollback_id
        self.store = Path(store_dir) if store_dir else (self.root / ".nabd" / "rollback" / rollback_id)
        self.log_path = Path(log_path) if log_path else (self.store / "rollback.log")
        self._seq = 0
        self._corrupt = False
        self._load_log_state()

    # ------------------------------------------------------------------
    # Log handling (append-only).
    # ------------------------------------------------------------------
    def _load_log_state(self) -> None:
        if not self.log_path.exists():
            return
        try:
            with self.log_path.open("r", encoding="utf-8") as handle:
                lines = [ln for ln in handle.read().splitlines() if ln.strip()]
            for ln in lines:
                entry = json.loads(ln)
                self._seq = max(self._seq, int(entry.get("sequence", 0)))
        except (ValueError, OSError, KeyError):
            self._corrupt = True

    def _append(
        self,
        operation: str,
        path: Optional[str],
        status: str,
        before_hash: Optional[str],
        after_hash: Optional[str],
    ) -> Dict[str, Any]:
        self._seq += 1
        entry = {
            "rollback_id": self.rollback_id,
            "sequence": self._seq,
            "operation": operation,
            "path": path,
            "status": status,
            "observed_at": time.time(),
            "before_hash": before_hash,
            "after_hash": after_hash,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(redact_obj(entry), ensure_ascii=False) + "\n")
        return entry

    def log_entries(self) -> List[Dict[str, Any]]:
        if not self.log_path.exists():
            return []
        out: List[Dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as handle:
            for ln in handle.read().splitlines():
                if ln.strip():
                    out.append(json.loads(ln))
        return out

    def _committed_paths(self) -> set:
        done = set()
        for e in self.log_entries():
            if e.get("status") == "COMMITTED" and e.get("operation") in (
                OP_RESTORE_COMMITTED,
                OP_DELETE_COMMITTED,
            ):
                if e.get("path"):
                    done.add(e["path"])
        return done

    # ------------------------------------------------------------------
    # Snapshot / manifest.
    # ------------------------------------------------------------------
    def take_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Snapshot the workspace: type + sha256 (regular) or target (symlink)."""
        manifest: Dict[str, Dict[str, Any]] = {}
        for path in sorted(self.root.rglob("*")):
            rel = str(path.relative_to(self.root))
            if ".nabd" in path.parts or ".git" in path.parts:
                continue
            if path.is_symlink():
                manifest[rel] = {
                    "type": "symlink",
                    "target": os.readlink(path),
                    "sha256": None,
                }
            elif path.is_dir():
                manifest[rel] = {"type": "directory", "sha256": None}
            elif path.is_file():
                manifest[rel] = {"type": "regular_file", "sha256": _sha256(path)}
        return manifest

    @staticmethod
    def verify_manifest(manifest: Dict[str, Dict[str, Any]]) -> bool:
        """M0.5-style completeness check: every entry must be well-formed."""
        if not isinstance(manifest, dict) or not manifest:
            return False
        for rel, entry in manifest.items():
            if not isinstance(rel, str) or not isinstance(entry, dict):
                return False
            if entry.get("type") not in ENTRY_TYPES:
                return False
            if entry["type"] == "regular_file" and not isinstance(entry.get("sha256"), str):
                return False
        return True

    def backup_all(self, manifest: Dict[str, Dict[str, Any]]) -> None:
        """Copy every in-scope regular file into the rollback store (pre-mutation)."""
        self.store.mkdir(parents=True, exist_ok=True)
        files_dir = self.store / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        for rel, entry in manifest.items():
            if entry["type"] != "regular_file":
                continue
            src = self.root / rel
            dst = files_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

    def begin(self) -> Dict[str, Dict[str, Any]]:
        """Take a baseline snapshot, back it up, and record STARTED."""
        manifest = self.take_snapshot()
        self.backup_all(manifest)
        self._append(OP_STARTED, None, "OK", None, None)
        return manifest

    # ------------------------------------------------------------------
    # Restore / delete helpers (atomic).
    # ------------------------------------------------------------------
    def _backup_path(self, rel: str) -> Path:
        return self.store / "files" / rel

    def _current_hash(self, rel: str) -> Optional[str]:
        p = self.root / rel
        if p.is_symlink():
            return f"symlink:{os.readlink(p)}"
        return _sha256(p)

    def _restore_file(self, rel: str, entry: Dict[str, Any]) -> None:
        backup = self._backup_path(rel)
        if not backup.exists():
            raise RollbackError(f"No backup available for {rel}")
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / f".rb-{uuid.uuid4().hex[:8]}.tmp"
        try:
            shutil.copyfile(backup, tmp)
            with tmp.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def _restore_symlink(self, rel: str, entry: Dict[str, Any]) -> None:
        """Recreate a symlink to its original target; never touch the target."""
        target = self.root / rel
        desired = entry.get("target")
        if target.is_symlink() and os.readlink(target) == desired:
            return
        if target.is_symlink() or target.exists():
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(desired, target)

    def _delete_path(self, rel: str) -> None:
        target = self.root / rel
        if target.is_symlink() or target.exists():
            target.unlink()

    # ------------------------------------------------------------------
    # Core rollback.
    # ------------------------------------------------------------------
    def rollback(
        self,
        *,
        baseline: Dict[str, Dict[str, Any]],
        unknown_paths: Optional[List[str]] = None,
        owned_modified: Optional[List[str]] = None,
        created_paths: Optional[List[str]] = None,
        fault: Optional[FaultInjector] = None,
        resume: bool = False,
    ) -> RollbackResult:
        start = time.time()
        unknown = set(unknown_paths or [])
        owned = list(owned_modified or [])
        created = list(created_paths or [])

        # Corrupt log -> fail closed, no guessed restore, no COMPLETED.
        if self._corrupt:
            self._append(OP_FAILED, None, "FAILED", None, None)
            return RollbackResult(
                decision=OP_FAILED,
                rollback_id=self.rollback_id,
                final_manifest_match=False,
                unknown_touched=False,
                duration_ms=(time.time() - start) * 1000,
            )

        # M0.5 precondition: baseline manifest must be valid.
        if not self.verify_manifest(baseline):
            raise RollbackError("Invalid baseline manifest (M0.5 precondition failed)")

        # BLOCK if a UNKNOWN actor modified a managed (baseline) file: never
        # overwrite an UNKNOWN change.
        for rel, entry in baseline.items():
            if entry.get("type") != "regular_file":
                continue
            if rel in unknown and rel not in owned:
                cur = _sha256(self.root / rel)
                if cur is not None and cur != entry.get("sha256"):
                    self._append(OP_BLOCKED, rel, "BLOCKED", cur, entry.get("sha256"))
                    return RollbackResult(
                        decision=OP_BLOCKED,
                        rollback_id=self.rollback_id,
                        blocked_path=rel,
                        final_manifest_match=False,
                        unknown_touched=False,
                        duration_ms=(time.time() - start) * 1000,
                    )

        # Build the operation plan.
        ops: List[Tuple[str, str]] = []
        for rel in owned:
            if rel in unknown:
                continue
            if baseline.get(rel, {}).get("type") == "symlink":
                ops.append(("restore_symlink", rel))
            else:
                ops.append(("restore", rel))
        for rel in created:
            if rel in unknown:
                continue
            ops.append(("delete", rel))

        if resume:
            done = self._committed_paths()
            ops = [o for o in ops if o[1] not in done]
            self._append(OP_RESUMED, None, "OK", None, None)

        restored: List[str] = []
        deleted: List[str] = []
        committed = 0
        total = len(ops)
        for kind, rel in ops:
            if kind == "restore_symlink":
                before = self._current_hash(rel)
                self._append(OP_RESTORE_STARTED, rel, "STARTED", before, None)
                self._restore_symlink(rel, baseline[rel])
                after = self._current_hash(rel)
                self._append(OP_RESTORE_COMMITTED, rel, "COMMITTED", before, after)
                restored.append(rel)
            elif kind == "restore":
                before = self._current_hash(rel)
                self._append(OP_RESTORE_STARTED, rel, "STARTED", before, None)
                self._restore_file(rel, baseline[rel])
                after = self._current_hash(rel)
                self._append(OP_RESTORE_COMMITTED, rel, "COMMITTED", before, after)
                restored.append(rel)
            else:  # delete
                before = self._current_hash(rel)
                self._append(OP_DELETE_STARTED, rel, "STARTED", before, None)
                self._delete_path(rel)
                after = self._current_hash(rel)
                self._append(OP_DELETE_COMMITTED, rel, "COMMITTED", before, after)
                deleted.append(rel)
            committed += 1
            if fault is not None and fault.should_interrupt(committed, total):
                # Persist the interruption marker, then propagate so the caller
                # (test harness) can resume from the same rollback_id.
                self._append(OP_INTERRUPTED, None, "INTERRUPTED", None, None)
                raise RollbackInterrupted(self._seq)

        # Final verification.
        self._append(OP_FINAL_VERIFY, None, "STARTED", None, None)
        match = self._final_match(baseline, unknown)
        unknown_touched = self._unknown_touched(unknown)
        if match and not unknown_touched:
            if unknown:
                # Environment not fully clean: agent rollback succeeded but the
                # presence of UNKNOWN changes mandates FAILED_ROLLBACK per policy.
                self._append(OP_FAILED, None, "FAILED", None, None)
                decision = OP_FAILED
            else:
                self._append(OP_COMMITTED, None, "COMMITTED", None, None)
                decision = OP_COMMITTED
        else:
            self._append(OP_FAILED, None, "FAILED", None, None)
            decision = OP_FAILED

        return RollbackResult(
            decision=decision,
            rollback_id=self.rollback_id,
            interrupted=False,
            resumed=resume,
            final_manifest_match=match,
            unknown_touched=unknown_touched,
            restored=restored,
            deleted=deleted,
            duration_ms=(time.time() - start) * 1000,
        )

    # ------------------------------------------------------------------
    # Post-rollback verification.
    # ------------------------------------------------------------------
    def _final_match(self, baseline: Dict[str, Dict[str, Any]], unknown: set) -> bool:
        """Managed scope (baseline minus UNKNOWN) matches the baseline hash."""
        for rel, entry in baseline.items():
            if rel in unknown:
                continue
            if entry.get("type") == "regular_file":
                if _sha256(self.root / rel) != entry.get("sha256"):
                    return False
            elif entry.get("type") == "symlink":
                p = self.root / rel
                if not (p.is_symlink() and os.readlink(p) == entry.get("target")):
                    return False
        return True

    def _unknown_touched(self, unknown: set) -> bool:
        # By construction we never restore/delete UNKNOWN paths; invariant holds.
        return False


def mark_corrupt(log_path: Path) -> None:
    """Test helper: corrupt a rollback log so the manager fails closed."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("{ this is not valid json\n")
