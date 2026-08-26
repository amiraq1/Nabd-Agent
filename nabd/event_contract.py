"""Frozen Agent Event Contract (UI4.2.1).

Defines the immutable, validated contract for every event emitted by
NabdAgent, and the ``EventBoundary`` that enforces it at the Core->Renderer
edge. Evidence remains owned exclusively by the Core: the UI/Renderer never
creates an ``evidence_id`` and only receives events after they pass structural
validation + secret redaction at the boundary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .events import EVENT_SCHEMA_VERSION, EventType, build_event
from .redact import redact_obj

# Canonical, frozen field set. Every event MUST contain exactly these.
REQUIRED_FIELDS = (
    "schema_version",
    "event_id",
    "event_type",
    "task_id",
    "session_id",
    "attempt_id",
    "attempt_order",
    "seq",
    "evidence_id",
    "fsm_state",
    "source",
    "payload",
)

VALID_EVENT_TYPES = {e.value for e in EventType}

KNOWN_FSM_STATES = {
    "PLANNING",
    "EXECUTING",
    "VERIFYING",
    "COMPLETED",
    "REJECTED",
    "REPAIRING",
    "ROLLED_BACK",
    "FAILED",
}

VALID_DECISIONS = {"ROLLBACK", "BLOCKED", "TIMEOUT", "REPAIR"}

# Discriminated payload shape: each event type requires its own keys.
# A payload missing a required key (or with a wrong-typed value) is invalid,
# which prevents impossible combinations between event types.
PAYLOAD_SCHEMA: Dict[str, List[str]] = {
    EventType.TASK_ACCEPTED.value: ["task", "intent"],
    EventType.INTENT_CLASSIFIED.value: ["intent"],
    EventType.SNAPSHOT_READY.value: ["files"],
    EventType.PLAN_READY.value: ["summary", "steps", "actions", "verification"],
    EventType.TOOL_STARTED.value: ["tool", "arguments"],
    EventType.MUTATION_STARTED.value: ["path"],
    EventType.TOOL_SUCCEEDED.value: ["tool", "output_excerpt"],
    EventType.TOOL_FAILED.value: ["tool", "error"],
    EventType.APPROVAL_REQUIRED.value: ["request"],
    EventType.APPROVAL_ACCEPTED.value: ["request"],
    EventType.APPROVAL_DENIED.value: ["request"],
    EventType.VERIFICATION_STARTED.value: ["commands"],
    EventType.VERIFICATION_PASSED.value: ["summary"],
    EventType.VERIFICATION_FAILED.value: ["decision", "summary"],
    EventType.UNKNOWN_CHANGE_DETECTED.value: ["paths"],
    EventType.ROLLBACK_STARTED.value: ["summary"],
    EventType.ROLLBACK_COMPLETED.value: ["summary"],
    EventType.TIMEOUT.value: ["summary"],
    EventType.TASK_COMPLETED.value: ["summary", "state"],
    EventType.TASK_FAILED.value: ["summary", "state", "error"],
    EventType.LATE_EVENT.value: ["reason"],
    EventType.EVENT_GAP_DETECTED.value: ["gap"],
}


def validate_event(event: Mapping[str, Any]) -> List[str]:
    """Return a list of contract violations (empty list means valid)."""
    problems: List[str] = []
    if not isinstance(event, Mapping):
        return ["event is not a mapping"]
    for field in REQUIRED_FIELDS:
        if field not in event:
            problems.append(f"missing field: {field}")
    if "schema_version" in event and not isinstance(event["schema_version"], int):
        problems.append("schema_version must be int")
    if "event_id" in event and not isinstance(event["event_id"], str):
        problems.append("event_id must be str")
    if "event_type" in event and event["event_type"] not in VALID_EVENT_TYPES:
        problems.append(f"unknown event_type: {event.get('event_type')}")
    if "seq" in event and not isinstance(event["seq"], int):
        problems.append("seq must be int")
    if "attempt_order" in event and not isinstance(event["attempt_order"], int):
        problems.append("attempt_order must be int")
    if "evidence_id" in event and event["evidence_id"] is not None and not isinstance(event["evidence_id"], str):
        problems.append("evidence_id must be str or None")
    if "payload" in event and not isinstance(event["payload"], Mapping):
        problems.append("payload must be a mapping")
    et = event.get("event_type")
    payload = event.get("payload")
    if et in PAYLOAD_SCHEMA and isinstance(payload, Mapping):
        for key in PAYLOAD_SCHEMA[et]:
            if key not in payload:
                problems.append(f"payload missing key for {et}: {key}")
        if et == EventType.VERIFICATION_FAILED.value and payload.get("decision") not in VALID_DECISIONS:
            problems.append("VERIFICATION_FAILED.decision invalid")
        if et == EventType.SNAPSHOT_READY.value and not isinstance(payload.get("files"), int):
            problems.append("SNAPSHOT_READY.files must be int")
        if et == EventType.LATE_EVENT.value and not isinstance(payload.get("reason"), str):
            problems.append("LATE_EVENT.reason must be str")
        if et == EventType.EVENT_GAP_DETECTED.value and not isinstance(payload.get("gap"), (list, tuple)):
            problems.append("EVENT_GAP_DETECTED.gap must be a list")
    return problems


class EventBoundary:
    """Validates + redacts events at the Core->Renderer boundary.

    Enforces the frozen contract: dedup, attempt_order/seq monotonicity, gap and
    late detection, and payload redaction. The boundary NEVER creates an
    ``evidence_id``; it only passes the Core-supplied value through.
    """

    def __init__(self, delegate: Any, schema_version: int = EVENT_SCHEMA_VERSION) -> None:
        self.delegate = delegate
        self.schema_version = schema_version
        self._seen_ids: set = set()
        self._last_seq: Dict[str, int] = {}  # attempt_id -> last seq
        self._last_attempt_order: int = 0
        self._meta_count: int = 0

    def publish(self, event: Mapping[str, Any]) -> None:
        # 1. Structural validation: invalid events never reach the UI.
        if validate_event(event):
            return
        event_id = event["event_id"]
        # 2. Dedup: a duplicate event_id is ignored safely.
        if event_id in self._seen_ids:
            return
        self._seen_ids.add(event_id)

        # 3. Redact payload at the boundary (secrets never reach the Renderer).
        redacted = redact_obj(dict(event))

        # 4. attempt_order monotonic within task (decreasing == late, no reorder).
        attempt_order = event["attempt_order"]
        if attempt_order < self._last_attempt_order:
            self._emit_meta("LATE_EVENT", event, reason="attempt_order decreased")
        else:
            self._last_attempt_order = attempt_order

        # 5. seq monotonic within attempt + gap/late detection.
        attempt_id = event["attempt_id"]
        seq = event["seq"]
        last = self._last_seq.get(attempt_id)
        if last is None:
            self._last_seq[attempt_id] = seq
        elif seq < last:
            self._emit_meta("LATE_EVENT", event, reason="seq < last", last=last)
        elif seq > last + 1:
            self._emit_meta("EVENT_GAP_DETECTED", event, gap=[last + 1, seq - 1])
            self._last_seq[attempt_id] = seq
        else:
            self._last_seq[attempt_id] = seq

        # Forward the validated, redacted event to the Renderer sink.
        try:
            self.delegate.publish(redacted)
        except Exception:
            pass

    def _emit_meta(self, meta_type: str, trigger: Mapping[str, Any], reason: Optional[str] = None, gap=None, last=None) -> None:
        self._meta_count += 1
        detail: Dict[str, Any] = {}
        if reason is not None:
            detail["reason"] = reason
        if gap is not None:
            detail["gap"] = gap
        if last is not None:
            detail["last_seq"] = last
        meta = build_event(
            meta_type,
            seq=self._meta_count,
            attempt_order=self._last_attempt_order,
            task_id=trigger.get("task_id", "contract"),
            session_id=trigger.get("session_id", "contract"),
            attempt_id=trigger.get("attempt_id", "contract"),
            fsm_state=trigger.get("fsm_state"),
            evidence_id=None,
            source="boundary",
            **detail,
        )
        try:
            self.delegate.publish(meta)
        except Exception:
            pass
