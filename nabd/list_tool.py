"""Raw-facts, jail-checked file listing."""

from __future__ import annotations

from pathlib import Path

from .jail import JailError, WorkspaceJail
from .raw_facts import RawFacts


class ListTool:
    """List at most 200 files and explicitly report truncation."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.root = Path(workspace_root).expanduser().resolve()
        self.jail = WorkspaceJail(self.root)

    def run(self, subpath: str = ".") -> RawFacts:
        target = self.jail.check_path(subpath, allow_missing=False)
        if not target.is_dir():
            return RawFacts(operation="list", path=subpath, status="TOOL_ERROR", error=f"Not a directory: {subpath}")
        files: list[str] = []
        saw_more = False
        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            # Hide Nabd's own runtime artifacts so they never enter model context.
            if ".nabd" in path.parts:
                continue
            try:
                safe_path = self.jail.check_path(path, allow_missing=False)
            except JailError:
                continue
            if len(files) >= 200:
                saw_more = True
                break
            files.append(str(safe_path.relative_to(self.root)))
        return RawFacts(
            operation="list",
            path=str(target.relative_to(self.root) or "."),
            exists=True,
            details={"files": files, "count": len(files), "truncated": saw_more},
            truncated=saw_more,
        )
