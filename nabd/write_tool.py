"""Raw-facts file writer; evidence verification belongs to EvidenceStore."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from .jail import WorkspaceJail
from .raw_facts import RawFacts


class WriteTool:
    """Write inside the workspace and return an untrusted filesystem receipt."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.root = Path(workspace_root).expanduser().resolve()
        self.jail = WorkspaceJail(self.root)

    def run(self, path: str, content: str) -> RawFacts:
        target = self.jail.check_path(path, allow_missing=True)
        backup: str | None = None
        if target.exists():
            backup_dir = self.jail.check_path(".nabd/backups", allow_missing=True)
            backup_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup_path = backup_dir / f"{target.name}.backup.{timestamp}"
            shutil.copy2(target, backup_path)
            backup = str(backup_path.relative_to(self.root))
        sha256 = self.jail.safe_write(target, content)
        stat = target.stat()
        return RawFacts(
            operation="write",
            path=str(target.relative_to(self.root)),
            exists=target.is_file(),
            size=stat.st_size,
            sha256=sha256,
            mtime=stat.st_mtime,
            backup=backup,
            details={"content_bytes": len(content.encode("utf-8"))},
        )
