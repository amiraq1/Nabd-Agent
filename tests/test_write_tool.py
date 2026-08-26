import tempfile
import unittest
from pathlib import Path

from nabd.evidence import EvidenceStore, EvidenceType
from nabd.raw_facts import RawFacts
from nabd.write_tool import WriteTool


class WriteToolTests(unittest.TestCase):
    def test_run_returns_raw_facts_without_minting_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = EvidenceStore(root, task_id="task-1")
            facts = WriteTool(root).run("hello.txt", "hello")
            self.assertIsInstance(facts, RawFacts)
            self.assertEqual(len(facts.sha256 or ""), 64)
            self.assertEqual(store.get_all(), [])
            evidence = store.verify(facts, task_id="task-1")
            self.assertEqual(evidence.evidence_type, EvidenceType.OBSERVED)
            self.assertTrue(store.is_usable_for_completion("task-1"))

    def test_run_backs_up_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "hello.txt"
            target.write_text("old", encoding="utf-8")
            facts = WriteTool(root).run("hello.txt", "new")
            self.assertIsNotNone(facts.backup)
            backups = list((root / ".nabd" / "backups").glob("hello.txt.*.bak"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), "old")


if __name__ == "__main__":
    unittest.main()
