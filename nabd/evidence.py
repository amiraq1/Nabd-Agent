"""Evidence verification for trustworthy task completion."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .raw_facts import RawFacts


class EvidenceType(Enum):
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"


@dataclass
class Evidence:
    claim: str
    evidence_type: EvidenceType
    task_id: str = "legacy"
    operation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    path: Optional[str] = None
    sha256: Optional[str] = None
    observed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    relevant: bool = True
    fresh: bool = True
    valid: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def timestamp(self) -> str:
        """Compatibility alias for the former timestamp field."""
        return self.observed_at

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["type"] = self.evidence_type.value
        del value["evidence_type"]
        return value


class EvidenceStore:
    """Verifies raw tool receipts and is the only issuer of OBSERVED evidence."""

    def __init__(self, root: str | Path, task_id: str = "legacy") -> None:
        self.root = Path(root).expanduser().resolve()
        self.task_id = task_id
        self._evidence: List[Evidence] = []

    @staticmethod
    def new_task_id() -> str:
        return f"task-{uuid.uuid4().hex}"

    def _safe_file(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("Evidence path escapes the project root")
        if not candidate.is_file():
            raise FileNotFoundError(f"Cannot verify file: {path} does not exist")
        return candidate

    @staticmethod
    def compute_file_hash(path: str | Path) -> str:
        target = Path(path).expanduser()
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def verify(
        self,
        raw_facts: RawFacts,
        claim: Optional[str] = None,
        task_id: Optional[str] = None,
        relevant: bool = True,
        max_age_seconds: Optional[float] = 300.0,
    ) -> Evidence:
        """Re-read raw facts and mint OBSERVED only when every predicate passes."""
        requested_task = task_id or self.task_id
        details = dict(raw_facts.details)
        details.update(
            {
                "operation": raw_facts.operation,
                "operation_id": raw_facts.operation_id,
                "raw_facts": raw_facts.to_dict(),
            }
        )
        valid = raw_facts.successful
        fresh = True
        relative: Optional[str] = None
        actual_hash: Optional[str] = None
        if raw_facts.path and (raw_facts.operation in {"read", "write", "file", "observe"} or raw_facts.operation.startswith("observe ")):
            try:
                target = self._safe_file(raw_facts.path)
                relative = str(target.relative_to(self.root))
                actual_hash = self.compute_file_hash(target)
                valid = valid and raw_facts.exists and raw_facts.sha256 == actual_hash
                if raw_facts.size is not None:
                    valid = valid and raw_facts.size == target.stat().st_size
            except (OSError, ValueError):
                valid = False
            if raw_facts.sha256 and actual_hash:
                fresh = actual_hash == raw_facts.sha256
        if max_age_seconds is not None and raw_facts.mtime is not None:
            age = datetime.now(timezone.utc).timestamp() - raw_facts.mtime
            fresh = fresh and age <= max_age_seconds
        valid = valid and requested_task == self.task_id
        evidence_type = EvidenceType.OBSERVED if valid and fresh and relevant else EvidenceType.INFERRED
        details.update(
            {
                "actual_sha256": actual_hash,
                "verification": "re-read filesystem",
                "reason": "verified" if evidence_type is EvidenceType.OBSERVED else "raw facts did not satisfy verification predicates",
            }
        )
        evidence = Evidence(
            claim=claim or raw_facts.operation,
            evidence_type=evidence_type,
            task_id=requested_task,
            operation_id=raw_facts.operation_id,
            path=relative,
            sha256=actual_hash or raw_facts.sha256,
            relevant=relevant,
            fresh=fresh,
            valid=valid,
            details=details,
        )
        self._evidence.append(evidence)
        return evidence

    def add_observed(
        self,
        claim: str,
        path: str,
        details: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> Evidence:
        """Compatibility wrapper: construct raw facts, then verify them."""
        target = self._safe_file(path)
        stat = target.stat()
        raw = RawFacts(
            operation=f"observe {path}",
            path=path,
            exists=True,
            size=stat.st_size,
            sha256=self.compute_file_hash(target),
            mtime=stat.st_mtime,
            details=details or {},
        )
        return self.verify(raw, claim=claim, task_id=task_id)

    def add_observed_check(
        self,
        claim: str,
        command: str,
        exit_code: int,
        output: str = "",
        task_id: Optional[str] = None,
    ) -> Evidence:
        if exit_code != 0:
            raise ValueError("Only successful checks may be marked OBSERVED")
        raw = RawFacts(
            operation=command,
            exit_code=exit_code,
            stdout=output[-4000:],
            details={"kind": "verification_command", "command": command},
        )
        return self.verify(raw, claim=claim, task_id=task_id)

    def add_inferred(self, claim: str, details: Optional[Dict[str, Any]] = None, task_id: Optional[str] = None) -> Evidence:
        evidence = Evidence(
            claim=claim,
            evidence_type=EvidenceType.INFERRED,
            task_id=task_id or self.task_id,
            relevant=False,
            fresh=False,
            valid=False,
            details=details or {},
        )
        self._evidence.append(evidence)
        return evidence

    def get_all(self) -> List[Evidence]:
        return list(self._evidence)

    def get_observed(self) -> List[Evidence]:
        return [item for item in self._evidence if item.evidence_type is EvidenceType.OBSERVED]

    def get_inferred(self) -> List[Evidence]:
        return [item for item in self._evidence if item.evidence_type is EvidenceType.INFERRED]

    def is_usable_for_completion(self, task_id: str, max_age_seconds: Optional[float] = 300.0) -> bool:
        observed = self.get_observed()
        if not observed or self.get_inferred():
            return False
        if any(
            item.task_id != task_id or not item.relevant or not item.fresh or not item.valid
            for item in observed
        ):
            return False
        for item in observed:
            if item.path:
                try:
                    target = self._safe_file(item.path)
                    if self.compute_file_hash(target) != item.sha256:
                        return False
                except (OSError, ValueError):
                    return False
            if max_age_seconds is not None:
                try:
                    age = datetime.now(timezone.utc) - datetime.fromisoformat(item.observed_at)
                    if age.total_seconds() > max_age_seconds:
                        return False
                except ValueError:
                    return False
        return True

    def all_observed(self) -> bool:
        """Compatibility check; v2 callers should use is_usable_for_completion."""
        return self.is_usable_for_completion(self.task_id, max_age_seconds=None)

    def save(self, relative_path: str = ".nabd/evidence.json") -> Path:
        target = (self.root / relative_path).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError("Evidence output escapes the project root")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "task_id": self.task_id,
            "evidence": [item.to_dict() for item in self._evidence],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target
