import tempfile
import unittest
from pathlib import Path

from nabd.approval import ApprovalMode
from nabd.events import EventType
from nabd.runtime import run_task

from helpers.fake_llm import FakeLLMClient


class Ui4EvidenceReadThroughTests(unittest.TestCase):
    def test_completed_event_reads_evidence_id_from_core_store(self):
        with tempfile.TemporaryDirectory(prefix="ui4-evidence-") as directory:
            root = Path(directory)
            (root / "README.md").write_text("original\n", encoding="utf-8")
            plan = {
                "summary": "evidence read-through",
                "steps": ["read", "write", "verify"],
                "actions": [
                    {"name": "read_file", "arguments": {"path": "README.md"}},
                    {"name": "write_file", "arguments": {"path": "README.md", "content": "updated\n"}},
                ],
                "verification": ["true"],
            }
            result = run_task(
                "عدّل README",
                root=root,
                provider="openai",
                llm_client=FakeLLMClient(root, plan_override=plan),
                approval_callback=lambda _request: True,
                approval_mode=ApprovalMode.CONFIRM,
            )

            self.assertTrue(result.ok)
            completed = next(
                event for event in result.events
                if event["event_type"] == EventType.TASK_COMPLETED.value
            )
            evidence_ids = {item["operation_id"] for item in result.evidence}
            self.assertIsNotNone(completed["evidence_id"])
            self.assertIn(completed["evidence_id"], evidence_ids)
            self.assertIn(completed["evidence_id"], completed["payload"]["evidence_ids"])

    def test_adapter_does_not_mint_missing_evidence_id(self):
        from nabd.events import EventType, build_event
        from nabd.ui_adapter import EventAdapter

        event = build_event(
            EventType.VERIFICATION_PASSED,
            seq=1,
            attempt_order=1,
            task_id="task-1",
            session_id="session-1",
            attempt_id="attempt-1",
            fsm_state="VERIFYING",
            evidence_id=None,
            source="core",
            summary="no evidence",
        )
        projected = EventAdapter().adapt(event)
        self.assertIsNone(projected.evidence_id)


if __name__ == "__main__":
    unittest.main()
