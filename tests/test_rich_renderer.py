import unittest

from rich.console import Console

from nabd.events import EventType, build_event
from nabd.rich_renderer import RichEventSink


class RichRendererTests(unittest.TestCase):
    def test_sink_routes_core_event_through_boundary_and_adapter(self):
        console = Console(record=True, width=100, color_system=None)
        sink = RichEventSink(console=console, live=False)
        event = build_event(
            EventType.TASK_ACCEPTED,
            seq=1,
            attempt_order=1,
            task_id="task-1",
            session_id="session-1",
            attempt_id="attempt-1",
            fsm_state="PLANNING",
            evidence_id=None,
            source="core",
            task="inspect repository",
            intent="READ_ONLY",
        )
        sink.publish(event)
        self.assertEqual(len(sink.renderer.events), 1)
        self.assertEqual(sink.renderer.events[0].event_type, "TASK_ACCEPTED")
        self.assertEqual(len(sink.canonical_log), 1)
        self.assertIsNone(sink.canonical_log.snapshot()[0]["evidence_id"])

    def test_renderer_is_display_only_and_secret_is_not_rendered(self):
        console = Console(record=True, width=100, color_system=None)
        sink = RichEventSink(console=console, live=False)
        secret = "api_key=render-secret"
        event = build_event(
            EventType.TOOL_FAILED,
            seq=1,
            attempt_order=1,
            task_id="task-1",
            session_id="session-1",
            attempt_id="attempt-1",
            fsm_state="FAILED",
            evidence_id="evidence-1",
            source="core",
            tool="run_command",
            error=secret,
        )
        sink.publish(event)
        rendered = console.export_text()
        # No live print occurs unless requested; full render remains safe.
        console.print(sink.render())
        rendered = console.export_text()
        self.assertNotIn("render-secret", rendered)
        self.assertIn("[REDACTED]", rendered)
        self.assertIn("evidence-1", rendered)

    def test_invalid_event_never_reaches_renderer(self):
        sink = RichEventSink(console=Console(record=True, color_system=None))
        invalid = {
            "schema_version": 99,
            "event_id": "bad",
            "event_type": "TASK_FAILED",
            "task_id": "task-1",
            "session_id": "session-1",
            "attempt_id": "attempt-1",
            "attempt_order": 1,
            "seq": 1,
            "evidence_id": None,
            "fsm_state": "FAILED",
            "source": "core",
            "payload": {"summary": "bad", "state": "FAILED", "error": "bad"},
        }
        sink.publish(invalid)
        self.assertEqual(sink.renderer.events, [])
        self.assertEqual(len(sink.canonical_log), 0)


if __name__ == "__main__":
    unittest.main()
