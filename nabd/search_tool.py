"""Raw-facts project search with explicit result status."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .jail import JailError, WorkspaceJail
from .raw_facts import RawFacts


class SearchTool:
    """Search fixed text safely and report backend/result details as raw facts."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.root = Path(workspace_root).expanduser().resolve()
        self.jail = WorkspaceJail(self.root)

    def run(self, query: str, subpath: str = ".", max_results: int = 50) -> RawFacts:
        query = query.strip()
        if not query:
            return RawFacts(operation="search", status="TOOL_ERROR", error="Search query cannot be empty")
        limit = max(1, min(int(max_results), 50))
        target = self.jail.check_path(subpath, allow_missing=False)
        if not target.is_dir():
            return RawFacts(operation="search", path=subpath, status="TOOL_ERROR", error=f"Not a directory: {subpath}")

        matches, backend, rg_error = self._search_rg(query, target, limit)
        if matches is None:
            matches = self._search_python(query, target, limit)
            backend = "python_fallback"
        details = {
            "matches": matches,
            "count": len(matches),
            "result": "MATCH" if matches else "NO_MATCH",
            "backend": backend,
            "fallback_used": backend == "python_fallback",
        }
        if rg_error:
            details["rg_error"] = rg_error
        return RawFacts(
            operation="search",
            path=str(target.relative_to(self.root) or "."),
            exists=True,
            details=details,
        )

    def _search_rg(self, query: str, target: Path, limit: int) -> tuple[list[str] | None, str, str | None]:
        rg = shutil.which("rg")
        if not rg:
            return None, "python_fallback", "ripgrep unavailable"
        command = [
            rg, "--files-with-matches", "--fixed-strings", "--hidden",
            "--glob", "!.git/**", "--glob", "!.env", "--glob", "!.env.*",
            "--glob", "!.ssh/**", "--glob", "!.aws/**", "--glob", "!.config/**",
            query, str(target),
        ]
        try:
            result = subprocess.run(command, cwd=str(self.root), capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            return None, "python_fallback", str(exc)
        if result.returncode not in (0, 1):
            return None, "python_fallback", result.stderr.strip() or f"rg exit {result.returncode}"
        matches: list[str] = []
        for raw in result.stdout.splitlines():
            try:
                safe_path = self.jail.check_path(raw, allow_missing=False)
            except JailError:
                continue
            matches.append(str(safe_path.relative_to(self.root)))
            if len(matches) >= limit:
                break
        return matches, "ripgrep", None

    def _search_python(self, query: str, target: Path, limit: int) -> list[str]:
        matches: list[str] = []
        for path in sorted(target.rglob("*")):
            if len(matches) >= limit:
                break
            if not path.is_file():
                continue
            try:
                safe_path = self.jail.check_path(path, allow_missing=False)
                text = safe_path.read_text(encoding="utf-8", errors="ignore")
            except (JailError, OSError, UnicodeError):
                continue
            if query in text:
                matches.append(str(safe_path.relative_to(self.root)))
        return matches
