"""M3 — UNKNOWN Reconciliation and External Changes.

Classifies every filesystem delta observed between the pre-task snapshot and
the post-task snapshot into one of:

    AGENT_CREATED, AGENT_MODIFIED, AGENT_DELETED,
    UNKNOWN_CREATED, UNKNOWN_MODIFIED, UNKNOWN_DELETED, UNKNOWN_SYMLINK

A delta is AGENT-owned only when it is attributable to a ToolCall the agent
executed (passed in via ``agent_owned``). Everything else -- including an
externally injected symlink -- is UNKNOWN, and the agent must never roll it
back or write over it.

The typed snapshot is *symlink-safe*: it records a symlink's target without
ever following it (no ``open``/read of the target), so an injected symlink
pointing at ``/etc/passwd`` is detected without the agent reading that file.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Change categories.
AGENT_CREATED = "AGENT_CREATED"
AGENT_MODIFIED = "AGENT_MODIFIED"
AGENT_DELETED = "AGENT_DELETED"
AGENT_SYMLINK = "AGENT_SYMLINK"
UNKNOWN_CREATED = "UNKNOWN_CREATED"
UNKNOWN_MODIFIED = "UNKNOWN_MODIFIED"
UNKNOWN_DELETED = "UNKNOWN_DELETED"
UNKNOWN_SYMLINK = "UNKNOWN_SYMLINK"


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file() or path.is_symlink():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def typed_snapshot(root: Path) -> Dict[str, Dict[str, Any]]:
    """Snapshot the workspace recording entry TYPE (incl. symlink target).

    Symlinks are recorded as symlinks and NEVER followed: only ``os.readlink``
    and ``os.path.realpath`` (path resolution, no data read) are used, so an
    injected symlink to a sensitive file is detected without reading it.
    """
    root = Path(root).expanduser().resolve()
    manifest: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if ".nabd" in path.parts or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            target = os.readlink(path)
            try:
                outside = not str(Path(os.path.realpath(path))).startswith(str(root) + os.sep)
            except OSError:
                outside = True
            manifest[rel] = {
                "type": "symlink",
                "target": target,
                "sha256": None,
                "outside": outside,
            }
        elif path.is_dir():
            manifest[rel] = {"type": "directory", "sha256": None}
        elif path.is_file():
            manifest[rel] = {"type": "regular_file", "sha256": _sha256(path)}
    return manifest


def classify(
    before: Dict[str, Dict[str, Any]],
    after: Dict[str, Dict[str, Any]],
    agent_owned: Set[str],
) -> List[Dict[str, Any]]:
    """Return a structured record per changed path.

    ``agent_owned`` is the set of paths the agent acted on via ToolCalls.
    Anything changed that is NOT agent-owned is UNKNOWN. Parent directories of
    an agent-owned file are also treated as agent-owned, because creating a
    file implicitly creates its containing directory.
    """
    # Expand ownership to include parent directories of agent-owned files.
    expanded = set(agent_owned)
    for p in list(agent_owned):
        for parent in Path(p).parents:
            rel = str(parent)
            if rel in (".", ""):
                continue
            expanded.add(rel)
    records: List[Dict[str, Any]] = []
    for rel in sorted(set(before) | set(after)):
        b = before.get(rel)
        a = after.get(rel)
        owned = rel in expanded
        if owned:
            if b and not a:
                category = AGENT_DELETED
            elif not b and a:
                category = AGENT_CREATED
            elif b and a and b.get("sha256") != a.get("sha256"):
                category = AGENT_MODIFIED
            else:
                continue
        else:
            if b and not a:
                category = UNKNOWN_DELETED
            elif not b and a:
                category = UNKNOWN_CREATED
            elif b and a and b.get("sha256") != a.get("sha256"):
                category = UNKNOWN_MODIFIED
            else:
                continue
        # Symlinks are always treated by their link nature.
        if a is not None and a.get("type") == "symlink":
            category = AGENT_SYMLINK if owned else UNKNOWN_SYMLINK
        outside = bool(a.get("outside")) if (a and a.get("type") == "symlink") else False
        records.append(
            {
                "path": rel,
                "category": category,
                "agent_owned": owned,
                "symlink_outside_workspace": outside,
                # UNKNOWN deltas must be protected from rollback.
                "protected_from_rollback": category.startswith("UNKNOWN"),
            }
        )
    return records


def unknown_paths(records: List[Dict[str, Any]]) -> Set[str]:
    """The set of UNKNOWN-classified paths (must never be rolled back)."""
    return {r["path"] for r in records if r["category"].startswith("UNKNOWN")}


def compute_unknown(
    before: Dict[str, Dict[str, Any]],
    after: Dict[str, Dict[str, Any]],
    agent_owned: Set[str],
) -> Set[str]:
    """Convenience: directly return the UNKNOWN path set for a before/after pair."""
    return unknown_paths(classify(before, after, agent_owned))
