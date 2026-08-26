"""UI4.2.1 — Frozen Agent Event Contract tests.

These verify the immutable event contract and the EventBoundary enforcement:
valid events forward, missing/invalid events are rejected, duplicate event_ids
are ignored, late events are recorded without misleading reorder, sequence gaps
are surfaced as EVENT_GAP_DETECTED, and payloads are redacted before reaching
the Renderer. Evidence is never created at the boundary.
"""

from __future__ import annotations

import unittest

from nabd.events import EVENT_SCHEMA_VERSION, CapturingEventSink, EventType, build_event
from nabd.event_contract import (
    EventBoundary,
    PAYLOAD_SCHEMA,
    REQUIRED_FIELDS,
    validate_event,
)


def _valid_event(et, seq=1, attempt_order=1, attempt_id="attempt-1", payload=None, **over):
    ev = build_event(
        et,
        seq=seq,
        attempt_order=attempt_order,
        task_id="task-1",
        session_id="sess-1",
        attempt_id=attempt_id,
        fsm_state="EXECUTING",
        evidence_id=None,
        source="core",
        **(payload or {}),
    )
    ev.update(over)
    return ev


def test_valid_event_forwards():
    cap = CapturingEventSink()
    b = EventBoundary(cap)
    ev = _valid_event(EventType.TOOL_SUCCEEDED, payload={"tool": "read", "output_excerpt": "ok"})
    b.publish(ev)
    assert len(cap.events) == 1
    assert cap.events[0]["event_type"] == "TOOL_SUCCEEDED"
    assert cap.events[0]["schema_version"] == EVENT_SCHEMA_VERSION


def test_missing_field_rejected():
    cap = CapturingEventSink()
    b = EventBoundary(cap)
    ev = _valid_event(EventType.TOOL_SUCCEEDED, payload={"tool": "read", "output_excerpt": "ok"})
    del ev["event_id"]
    b.publish(ev)
    assert cap.events == []


def test_duplicate_event_id_ignored():
    cap = CapturingEventSink()
    b = EventBoundary(cap)
    ev = _valid_event(EventType.TOOL_SUCCEEDED, payload={"tool": "a", "output_excerpt": "b"})
    b.publish(ev)
    b.publish(dict(ev))  # same event_id
    assert len(cap.events) == 1


def test_late_event_recorded():
    cap = CapturingEventSink()
    b = EventBoundary(cap)
    b.publish(_valid_event(EventType.TOOL_SUCCEEDED, seq=3, payload={"tool": "a", "output_excerpt": "b"}))
    b.publish(_valid_event(EventType.TOOL_SUCCEEDED, seq=1, payload={"tool": "a", "output_excerpt": "b"}))
    types = [e["event_type"] for e in cap.events]
    assert "LATE_EVENT" in types
    # the late event is still forwarded (no misleading reorder)
    assert types.count("TOOL_SUCCEEDED") == 2


def test_sequence_gap_detected():
    cap = CapturingEventSink()
    b = EventBoundary(cap)
    b.publish(_valid_event(EventType.TOOL_SUCCEEDED, seq=1, payload={"tool": "a", "output_excerpt": "b"}))
    b.publish(_valid_event(EventType.TOOL_SUCCEEDED, seq=3, payload={"tool": "a", "output_excerpt": "b"}))
    types = [e["event_type"] for e in cap.events]
    assert "EVENT_GAP_DETECTED" in types
    gap_event = next(e for e in cap.events if e["event_type"] == "EVENT_GAP_DETECTED")
    assert gap_event["payload"]["gap"] == [2, 2]


def test_invalid_payload_rejected():
    cap = CapturingEventSink()
    b = EventBoundary(cap)
    # TOOL_SUCCEEDED requires both "tool" and "output_excerpt"
    b.publish(_valid_event(EventType.TOOL_SUCCEEDED, payload={"tool": "a"}))
    assert cap.events == []


def test_evidence_id_not_created_at_boundary():
    cap = CapturingEventSink()
    b = EventBoundary(cap)
    b.publish(_valid_event(EventType.TOOL_SUCCEEDED, payload={"tool": "a", "output_excerpt": "b"}, evidence_id=None))
    assert cap.events[0]["evidence_id"] is None
    # boundary never invents an evidence_id for any event it emits
    for e in cap.events:
        assert e["evidence_id"] is None


def test_attempt_order_decreasing_is_late():
    cap = CapturingEventSink()
    b = EventBoundary(cap)
    b.publish(_valid_event(EventType.PLAN_READY, attempt_order=1, payload={"summary": "s", "steps": [], "actions": [], "verification": []}))
    b.publish(_valid_event(EventType.PLAN_READY, attempt_order=0, payload={"summary": "s", "steps": [], "actions": [], "verification": []}))
    assert "LATE_EVENT" in [e["event_type"] for e in cap.events]


def test_redaction_at_boundary():
    cap = CapturingEventSink()
    b = EventBoundary(cap)
    ev = _valid_event(
        EventType.TOOL_FAILED,
        payload={"tool": "run_command", "error": "secret=supersecretvalue123 leaked"},
    )
    b.publish(ev)
    fwd = cap.events[0]
    assert "supersecretvalue123" not in fwd["payload"]["error"]
    assert "[REDACTED]" in fwd["payload"]["error"]


def test_contract_constants():
    assert EVENT_SCHEMA_VERSION == 1
    sample = _valid_event(EventType.TOOL_SUCCEEDED, payload={"tool": "x", "output_excerpt": "y"})
    for field in REQUIRED_FIELDS:
        assert field in sample
    # discriminated payload: every event type has its own required keys
    assert PAYLOAD_SCHEMA[EventType.TOOL_SUCCEEDED.value] == ["tool", "output_excerpt"]


def test_validate_event_helper():
    good = _valid_event(EventType.TOOL_SUCCEEDED, payload={"tool": "x", "output_excerpt": "y"})
    assert validate_event(good) == []
    bad = dict(good)
    bad["event_type"] = "NOT_A_REAL_TYPE"
    assert validate_event(bad)  # non-empty list of problems


def test_schema_is_closed_and_versioned():
    good = _valid_event(EventType.TOOL_SUCCEEDED, payload={"tool": "x", "output_excerpt": "y"})
    extra = dict(good)
    extra["new_field_without_version"] = True
    assert any("unknown fields" in problem for problem in validate_event(extra))
    wrong_version = dict(good)
    wrong_version["schema_version"] = EVENT_SCHEMA_VERSION + 1
    assert any("unsupported schema_version" in problem for problem in validate_event(wrong_version))


def test_contract_rejects_invalid_core_types_and_negative_sequence():
    bad = _valid_event(EventType.TOOL_SUCCEEDED, payload={"tool": "x", "output_excerpt": "y"})
    bad["seq"] = -1
    bad["attempt_order"] = "first"
    bad["payload"] = {"tool": "x", "output_excerpt": 123}
    problems = validate_event(bad)
    assert any("seq must be non-negative" in problem for problem in problems)
    assert any("attempt_order must be" in problem for problem in problems)
    assert any("TOOL_SUCCEEDED.output_excerpt must be str" in problem for problem in problems)


def load_tests(loader, _tests, _pattern):
    """Expose function-style contract tests to the project's unittest runner."""
    suite = unittest.TestSuite()
    for name in sorted(globals()):
        if name.startswith("test_") and callable(globals()[name]):
            suite.addTest(unittest.FunctionTestCase(globals()[name]))
    return suite
