"""Raw, untrusted facts returned by tools before evidence verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import uuid
from typing import Any, Dict, Optional


@dataclass
class RawFacts:
    """A factual tool receipt; it is never an OBSERVED evidence by itself."""

    operation: str
    operation_id: str = field(default_factory=lambda: f"op-{uuid.uuid4().hex}")
    path: Optional[str] = None
    exists: bool = False
    size: Optional[int] = None
    sha256: Optional[str] = None
    mtime: Optional[float] = None
    exit_code: Optional[int] = None
    truncated: bool = False
    backup: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    signal: Optional[int] = None
    attempt_id: str = ""
    sequence_number: int = 0
    status: str = "OK"
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def successful(self) -> bool:
        return self.status == "OK" and (self.exit_code is None or self.exit_code == 0) and self.error is None
