"""Display-only adapter from the core AgentEvent contract to immutable UiEvent.

The adapter is a presentation boundary. It validates the core-owned event,
redacts sensitive values, and projects a small display-safe object. It never
creates evidence IDs, evaluates verification, executes tools, or changes core
security decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Tuple

from .event_contract import validate_event
from .redact import redact_text


_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|password|secret|credential)"
)
_SENSITIVE_PATH_RE = re.compile(
    r"(?:^|[/\\])(?:\.env(?:\.[^/\\]+)?|\.ssh|\.aws|\.config)(?:[/\\]|$)",
    re.IGNORECASE,
)


def _redact_value(value: Any, key: Optional[str] = None) -> Any:
    if key and _SENSITIVE_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {str(k): _redact_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    return value


def _safe_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    path = redact_text(str(value))
    if _SENSITIVE_PATH_RE.search(path):
        return "[REDACTED_PATH]"
    return path


def _safe_paths(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    values: Iterable[Any] = (value,) if isinstance(value, str) else value
    return tuple(
        path
        for item in values
        if (path := _safe_path(item)) is not None
    )


@dataclass(frozen=True)
class UiEvent:
    """Immutable, display-safe projection of one core event."""

    schema_version: int
    event_id: str
    event_type: str
    task_id: str
    session_id: str
    attempt_id: Optional[str]
    attempt_order: int
    seq: int
    evidence_id: Optional[str]
    fsm_state: Optional[str]
    source: str
    summary: str = ""
    tool_name: Optional[str] = None
    file_path: Optional[str] = None
    decision: Optional[str] = None
    approval_required: bool = False
    approval_status: Optional[str] = None
    changed_files: Tuple[str, ...] = field(default_factory=tuple)
    unknown_paths: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.event_id or not self.task_id or not self.session_id:
            raise ValueError("event identity fields are required")
        if self.attempt_order < 0 or self.seq < 0:
            raise ValueError("event ordering values must be non-negative")

    @property
    def ordering_key(self) -> tuple[str, int, int]:
        """Canonical ordering key: task, attempt order, then sequence."""
        return (self.task_id, self.attempt_order, self.seq)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "attempt_id": self.attempt_id,
            "attempt_order": self.attempt_order,
            "seq": self.seq,
            "evidence_id": self.evidence_id,
            "fsm_state": self.fsm_state,
            "source": self.source,
            "summary": self.summary,
            "tool_name": self.tool_name,
            "file_path": self.file_path,
            "decision": self.decision,
            "approval_required": self.approval_required,
            "approval_status": self.approval_status,
            "changed_files": list(self.changed_files),
            "unknown_paths": list(self.unknown_paths),
            "metadata": dict(self.metadata),
        }


class EventAdapter:
    """Validate and project core events without changing their authority."""

    def adapt(self, raw: Mapping[str, Any]) -> UiEvent:
        problems = validate_event(raw)
        if problems:
            raise ValueError("invalid AgentEvent: " + "; ".join(problems))

        clean = _redact_value(dict(raw))
        payload = clean["payload"]
        if not isinstance(payload, Mapping):  # guarded by validate_event
            raise ValueError("AgentEvent payload must be a mapping")
        event_type = clean["event_type"]
        request = payload.get("request")
        request_map = request if isinstance(request, Mapping) else {}

        if event_type == "APPROVAL_REQUIRED":
            approval_status = "REQUIRED"
        elif event_type == "APPROVAL_ACCEPTED":
            approval_status = "ACCEPTED"
        elif event_type == "APPROVAL_DENIED":
            approval_status = "DENIED"
        else:
            approval_status = None

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        metadata_items = tuple(
            sorted((str(key), str(_redact_value(value, str(key)))) for key, value in metadata.items())
        )

        summary = payload.get("summary")
        if summary is None:
            summary = payload.get("error", payload.get("output_excerpt", ""))
        tool_name = payload.get("tool") or request_map.get("tool")
        return UiEvent(
            schema_version=clean["schema_version"],
            event_id=clean["event_id"],
            event_type=event_type,
            task_id=clean["task_id"],
            session_id=clean["session_id"],
            attempt_id=clean["attempt_id"],
            attempt_order=clean["attempt_order"],
            seq=clean["seq"],
            # Evidence ownership is strictly read-through: None stays None.
            evidence_id=clean["evidence_id"],
            fsm_state=clean["fsm_state"],
            source=clean["source"],
            summary=redact_text(str(summary or "")),
            tool_name=redact_text(str(tool_name)) if tool_name is not None else None,
            file_path=_safe_path(payload.get("path")),
            decision=redact_text(str(payload["decision"])) if "decision" in payload else None,
            approval_required=event_type == "APPROVAL_REQUIRED",
            approval_status=approval_status,
            changed_files=_safe_paths(payload.get("changed_files")),
            unknown_paths=_safe_paths(payload.get("paths", payload.get("unknown_paths"))),
            metadata=metadata_items,
        )


class UiEventStream:
    """In-memory display stream for deterministic tests and renderer binding."""

    def __init__(self, adapter: Optional[EventAdapter] = None) -> None:
        self.adapter = adapter or EventAdapter()
        self.events: list[UiEvent] = []

    def publish(self, raw: Mapping[str, Any]) -> UiEvent:
        event = self.adapter.adapt(raw)
        self.events.append(event)
        return event

    def snapshot(self) -> tuple[UiEvent, ...]:
        return tuple(self.events)
