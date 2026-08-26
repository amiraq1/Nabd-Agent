"""Crash-safe pre-run workspace snapshots for Nabd.

The snapshot is a typed manifest of workspace files. Internal artifacts,
repository metadata, bytecode caches, and environment files are excluded so
Nabd never hashes or persists credential-bearing files as part of this gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .jail import WorkspaceJail


SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_RELATIVE_PATH = ".nabd/snapshot-before.json"
_EXPECTED_KEYS = frozenset({"schema_version", "task_id", "root", "files"})
_FILE_KEYS = frozenset({"kind", "size", "mtime_ns", "sha256"})
_SYMLINK_KEYS = frozenset({"kind", "target", "mtime_ns"})
_EXCLUDED_DIRS = frozenset({".git", ".nabd", ".ssh", ".aws", ".config", "__pycache__"})


class SnapshotError(RuntimeError):
    """Raised when a snapshot cannot be created or validated."""


def _excluded(relative: Path) -> bool:
    if any(part in _EXCLUDED_DIRS for part in relative.parts):
        return True
    name = relative.name
    return name == ".env" or name.startswith(".env.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_entry(path: Path) -> Dict[str, Any]:
    stat = path.lstat()
    if path.is_symlink():
        return {
            "kind": "symlink",
            "target": os.readlink(path),
            "mtime_ns": stat.st_mtime_ns,
        }
    if not path.is_file():
        raise SnapshotError("workspace entry is neither a regular file nor a symlink")
    return {
        "kind": "file",
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(path),
    }


def build_manifest(root: Path, task_id: str) -> Dict[str, Any]:
    """Build a deterministic typed manifest without following symlinks."""
    if not root.is_dir():
        raise SnapshotError("workspace root does not exist")
    files: Dict[str, Dict[str, Any]] = {}
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        relative_dir = current_path.relative_to(root)
        dirnames[:] = sorted(
            name for name in dirnames
            if not _excluded(relative_dir / name)
        )
        for name in sorted(filenames):
            relative = (relative_dir / name) if relative_dir != Path(".") else Path(name)
            if _excluded(relative):
                continue
            path = root / relative
            try:
                files[str(relative)] = _manifest_entry(path)
            except (OSError, ValueError) as exc:
                raise SnapshotError("workspace entry could not be inspected") from exc
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "task_id": str(task_id),
        "root": str(root),
        "files": files,
    }


def validate_manifest(value: Any, root: Path, task_id: Optional[str] = None) -> bool:
    """Require an exact top-level key set and structurally valid entries."""
    if not isinstance(value, dict) or set(value) != set(_EXPECTED_KEYS):
        return False
    if value.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        return False
    if value.get("root") != str(root):
        return False
    if task_id is not None and value.get("task_id") != str(task_id):
        return False
    files = value.get("files")
    if not isinstance(files, dict):
        return False
    for relative, entry in files.items():
        relative_path = Path(relative) if isinstance(relative, str) else Path(".")
        if (
            not isinstance(relative, str)
            or not relative
            or relative_path.is_absolute()
            or any(part in {".", ".."} for part in relative_path.parts)
            or _excluded(relative_path)
        ):
            return False
        if not isinstance(entry, dict) or entry.get("kind") not in {"file", "symlink"}:
            return False
        expected = _FILE_KEYS if entry["kind"] == "file" else _SYMLINK_KEYS
        if set(entry) != set(expected):
            return False
        if not isinstance(entry.get("mtime_ns"), int):
            return False
        if entry["kind"] == "file":
            if not isinstance(entry.get("size"), int) or entry["size"] < 0:
                return False
            digest = entry.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                return False
        elif not isinstance(entry.get("target"), str):
            return False
    return True


class SnapshotStore:
    """Persist and reload a validated snapshot using an atomic replacement."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.jail = WorkspaceJail(self.root)

    @property
    def path(self) -> Path:
        return self.jail.check_internal_path(SNAPSHOT_RELATIVE_PATH, allow_missing=True)

    def load_if_valid(self, task_id: str) -> Optional[Dict[str, Any]]:
        target = self.path
        if not target.exists():
            return None
        try:
            value = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            target.unlink(missing_ok=True)
            return None
        if not validate_manifest(value, self.root, task_id):
            target.unlink(missing_ok=True)
            return None
        return value

    def create(self, task_id: str) -> Dict[str, Any]:
        manifest = build_manifest(self.root, task_id)
        if not validate_manifest(manifest, self.root, task_id):
            raise SnapshotError("snapshot manifest failed structural validation")
        target = self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=".snapshot-before.", dir=str(target.parent))
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (OSError, TypeError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            raise SnapshotError("snapshot manifest could not be persisted") from exc
        finally:
            temporary.unlink(missing_ok=True)
        return manifest

    def load_or_create(self, task_id: str) -> Dict[str, Any]:
        existing = self.load_if_valid(task_id)
        return existing if existing is not None else self.create(task_id)
