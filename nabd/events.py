"""Event stream + UI event adapter for NabdAgent.

The agent can optionally publish structured events to an ``AgentEventSink``
(e.g. a Rich renderer). Core behaviour is unchanged when no sink or approval
callback is provided. Every sink call is isolated, so a broken UI can never
crash the agent or bypass a safety policy.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Protocol


class EventType(str, Enum):
    TASK_ACCEPTED = "TASK_ACCEPTED"
    INTENT_CLASSIFIED = "INTENT_CLASSIFIED"
    SNAPSHOT_READY = "SNAPSHOT_READY"
    PLAN_READY = "PLAN_READY"
    TOOL_STARTED = "TOOL_STARTED"
    TOOL_SUCCEEDED = "TOOL_SUCCEEDED"
    TOOL_FAILED = "TOOL_FAILED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    APPROVAL_ACCEPTED = "APPROVAL_ACCEPTED"
    APPROVAL_DENIED = "APPROVAL_DENIED"
    MUTATION_STARTED = "MUTATION_STARTED"
    VERIFICATION_STARTED = "VERIFICATION_STARTED"
    VERIFICATION_PASSED = "VERIFICATION_PASSED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    UNKNOWN_CHANGE_DETECTED = "UNKNOWN_CHANGE_DETECTED"
    ROLLBACK_STARTED = "ROLLBACK_STARTED"
    ROLLBACK_COMPLETED = "ROLLBACK_COMPLETED"
    TIMEOUT = "TIMEOUT"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    # Boundary control events (emitted by the EventBoundary, not the Core).
    LATE_EVENT = "LATE_EVENT"
    EVENT_GAP_DETECTED = "EVENT_GAP_DETECTED"


# Frozen event schema version for UI4.2.1.
EVENT_SCHEMA_VERSION: int = 1


class AgentEventSink(Protocol):
    """Any object exposing ``publish(event)`` can receive the agent stream."""

    def publish(self, event: Mapping[str, Any]) -> None:  # pragma: no cover
        ...


def build_event(
    event_type: Any,
    *,
    seq: int,
    attempt_order: int,
    task_id: str,
    session_id: str,
    attempt_id: Optional[str] = None,
    fsm_state: Optional[str] = None,
    evidence_id: Optional[str] = None,
    source: str = "core",
    payload: Optional[Dict[str, Any]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a canonical, frozen-contract event dict (the EventAdapter output).

    The shape is immutable: every event contains exactly the contract fields
    (schema_version, event_id, event_type, task_id, session_id, attempt_id,
    attempt_order, seq, evidence_id, fsm_state, source, payload). Evidence
    ownership stays in the Core; ``evidence_id`` is never invented here.
    """
    if isinstance(event_type, Enum):
        event_type = event_type.value
    if payload is None:
        payload = {}
    else:
        payload = dict(payload)
    payload.update(extra)
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": uuid.uuid4().hex,
        "event_type": event_type,
        "task_id": task_id,
        "session_id": session_id,
        "attempt_id": attempt_id,
        "attempt_order": attempt_order,
        "seq": seq,
        "evidence_id": evidence_id,
        "fsm_state": fsm_state,
        "source": source,
        "payload": payload,
    }


class CapturingEventSink:
    """Collects every published event; used by tests and offline capture."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def publish(self, event: Mapping[str, Any]) -> None:
        self.events.append(dict(event))


class TeeEventSink:
    """Forwards to a delegate sink while capturing a local copy."""

    def __init__(self, delegate: Optional[AgentEventSink] = None) -> None:
        self.delegate = delegate
        self.events: List[Dict[str, Any]] = []

    def publish(self, event: Mapping[str, Any]) -> None:
        self.events.append(dict(event))
        if self.delegate is not None:
            self.delegate.publish(event)


class ConsoleEventSink:
    """Default human/renderer-friendly sink (one JSON line per event)."""

    def __init__(self, out: Optional[Any] = None) -> None:
        self._out = out

    def publish(self, event: Mapping[str, Any]) -> None:
        import json
        import sys

        print(
            json.dumps({"ui_event": dict(event)}, ensure_ascii=False),
            file=self._out or sys.stdout,
        )


@dataclass
class AgentRunResult:
    """Renderer-friendly result returned by ``run_task``."""

    state: str
    summary: str
    changes: List[str]
    evidence: List[Dict[str, Any]]
    error: Optional[str]
    ok: bool
    events: List[Dict[str, Any]] = field(default_factory=list)
