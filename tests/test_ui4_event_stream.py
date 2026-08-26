import unittest

from nabd.event_contract import EventBoundary
from nabd.event_stream import BoundedDisplayQueue, CanonicalEventLog
from nabd.events import CapturingEventSink, EventType, build_event


def make_event(seq=1, attempt_order=1, task_id="task-1", session_id="session-1", attempt_id="attempt-1"):
    return build_event(
        EventType.TOOL_SUCCEEDED,
        seq=seq,
        attempt_order=attempt_order,
        task_id=task_id,
        session_id=session_id,
        attempt_id=attempt_id,
        fsm_state="EXECUTING",
        evidence_id=None,
        source="core",
        tool="read",
        output_excerpt="ok",
    )


class Ui4EventStreamTests(unittest.TestCase):
    def test_canonical_log_is_deduplicated_and_keeps_arrival_order(self):
        capture = CapturingEventSink()
        log = CanonicalEventLog()
        boundary = EventBoundary(capture, canonical_log=log)
        raw = make_event(seq=1)
        boundary.publish(raw)
        boundary.publish(dict(raw))
        self.assertEqual(len(log), 1)
        self.assertEqual(log.snapshot()[0]["event_id"], raw["event_id"])

    def test_display_queue_is_bounded_but_canonical_log_keeps_all(self):
        capture = CapturingEventSink()
        log = CanonicalEventLog()
        queue = BoundedDisplayQueue(maxsize=2)
        boundary = EventBoundary(capture, canonical_log=log, display_queue=queue)
        for seq in range(1, 4):
            boundary.publish(make_event(seq=seq))
        self.assertEqual(len(log), 3)
        self.assertEqual(len(queue), 2)
        self.assertEqual(queue.dropped_count, 1)
        self.assertEqual([item["seq"] for item in queue.drain()], [2, 3])

    def test_late_event_is_recorded_without_reordering_canonical_arrival(self):
        capture = CapturingEventSink()
        log = CanonicalEventLog()
        boundary = EventBoundary(capture, canonical_log=log)
        boundary.publish(make_event(seq=2))
        boundary.publish(make_event(seq=1))
        types = [item["event_type"] for item in log.snapshot()]
        self.assertEqual(types, ["TOOL_SUCCEEDED", "LATE_EVENT", "TOOL_SUCCEEDED"])
        self.assertEqual(log.snapshot()[-1]["seq"], 1)

    def test_sequence_scope_includes_task_and_session(self):
        capture = CapturingEventSink()
        log = CanonicalEventLog()
        boundary = EventBoundary(capture, canonical_log=log)
        boundary.publish(make_event(seq=1, task_id="task-a", session_id="session-a"))
        boundary.publish(make_event(seq=3, task_id="task-a", session_id="session-a"))
        boundary.publish(make_event(seq=1, task_id="task-b", session_id="session-b"))
        boundary.publish(make_event(seq=2, task_id="task-a", session_id="session-a"))
        meta_types = [
            item["event_type"]
            for item in log.snapshot()
            if item["event_type"] in {"LATE_EVENT", "EVENT_GAP_DETECTED"}
        ]
        self.assertEqual(meta_types, ["EVENT_GAP_DETECTED", "LATE_EVENT"])

    def test_queue_rejects_non_positive_capacity(self):
        with self.assertRaises(ValueError):
            BoundedDisplayQueue(maxsize=0)


if __name__ == "__main__":
    unittest.main()
