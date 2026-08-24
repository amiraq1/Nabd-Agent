"""Raw-facts, jail-checked file reader."""

from __future__ import annotations

from pathlib import Path

from .jail import WorkspaceJail
from .raw_facts import RawFacts


class ReadTool:
    """Read a file and return raw content plus filesystem metadata."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.root = Path(workspace_root).expanduser().resolve()
        self.jail = WorkspaceJail(self.root)

    def run(self, path: str, max_bytes: int = 120_000) -> RawFacts:
        target = self.jail.check_path(path, allow_missing=False)
        if not target.is_file():
            return RawFacts(operation="read", path=path, status="TOOL_ERROR", error=f"Not a file: {path}")
        limit = max(1, min(int(max_bytes), 250_000))
        raw = target.read_bytes()
        truncated = len(raw) > limit
        content = raw[:limit].decode("utf-8", errors="replace")
        if truncated:
            content += "\n...[truncated]"
        stat = target.stat()
        return RawFacts(
            operation="read",
            path=str(target.relative_to(self.root)),
            exists=True,
            size=stat.st_size,
            sha256=self._hash_bytes(raw),
            mtime=stat.st_mtime,
            truncated=truncated,
            stdout=content,
        )

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        import hashlib
        return hashlib.sha256(data).hexdigest()
