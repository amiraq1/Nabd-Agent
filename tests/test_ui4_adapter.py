import unittest

from nabd.events import EventType, build_event
from nabd.ui_adapter import EventAdapter, UiEventStream


def event(event_type, payload, **kwargs):
    return build_event(
        event_type,
        seq=kwargs.pop("seq", 1),
        attempt_order=kwargs.pop("attempt_order", 1),
        task_id=kwargs.pop("task_id", "task-1"),
        session_id=kwargs.pop("session_id", "session-1"),
        attempt_id=kwargs.pop("attempt_id", "attempt-1"),
        fsm_state=kwargs.pop("fsm_state", "EXECUTING"),
        evidence_id=kwargs.pop("evidence_id", None),
        source="core",
        **payload,
        **kwargs,
    )


class Ui4AdapterTests(unittest.TestCase):
    def test_projects_core_event_and_preserves_ordering_key(self):
        projected = EventAdapter().adapt(
            event(
                EventType.TOOL_FAILED,
                {"tool": "run_command", "error": "failed"},
                seq=4,
                attempt_order=2,
                evidence_id="evidence-1",
            )
        )
        self.assertEqual(projected.ordering_key, ("task-1", 2, 4))
        self.assertEqual(projected.tool_name, "run_command")
        self.assertEqual(projected.summary, "failed")
        self.assertEqual(projected.evidence_id, "evidence-1")

    def test_invalid_core_event_is_rejected_before_projection(self):
        raw = event(EventType.TOOL_SUCCEEDED, {"tool": "read", "output_excerpt": "ok"})
        raw["schema_version"] = 99
        with self.assertRaises(ValueError):
            EventAdapter().adapt(raw)

    def test_recursive_redaction_happens_at_adapter_boundary(self):
        projected = EventAdapter().adapt(
            event(
                EventType.TOOL_FAILED,
                {
                    "tool": "run_command",
                    "error": "api_key=plain-secret Bearer bearer-secret",
                    "metadata": {"password": "plain-password", "note": "safe"},
                    "path": ".env.local",
                },
            )
        )
        self.assertNotIn("plain-secret", projected.summary)
        self.assertNotIn("bearer-secret", projected.summary)
        self.assertEqual(projected.file_path, "[REDACTED_PATH]")
        self.assertEqual(dict(projected.metadata)["password"], "[REDACTED]")
        self.assertEqual(dict(projected.metadata)["note"], "safe")

    def test_adapter_does_not_invent_evidence_id(self):
        projected = EventAdapter().adapt(
            event(EventType.TOOL_SUCCEEDED, {"tool": "read", "output_excerpt": "ok"})
        )
        self.assertIsNone(projected.evidence_id)

    def test_ui_event_is_immutable_and_stream_preserves_order(self):
        stream = UiEventStream()
        first = stream.publish(event(EventType.TOOL_STARTED, {"tool": "read", "arguments": {}}))
        second = stream.publish(
            event(EventType.TOOL_SUCCEEDED, {"tool": "read", "output_excerpt": "ok"}, seq=2)
        )
        self.assertEqual([item.seq for item in stream.snapshot()], [1, 2])
        with self.assertRaises((AttributeError, TypeError)):
            first.seq = 9
        self.assertEqual(second.ordering_key, ("task-1", 1, 2))


if __name__ == "__main__":
    unittest.main()
