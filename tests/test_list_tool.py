import tempfile
import unittest
from pathlib import Path

from nabd.list_tool import ListTool
from nabd.raw_facts import RawFacts


class ListToolTests(unittest.TestCase):
    def test_lists_safe_files_recursively(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "main.py").write_text("print(1)", encoding="utf-8")
            (root / ".env").write_text("SECRET=x", encoding="utf-8")
            facts = ListTool(root).run()
            self.assertIsInstance(facts, RawFacts)
            self.assertEqual(facts.details["files"], ["src/main.py"])
            self.assertEqual(facts.details["count"], 1)
            self.assertFalse(facts.truncated)

    def test_limits_output_to_200_files_and_reports_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(250):
                (root / f"file_{index:03d}.txt").write_text("x", encoding="utf-8")
            facts = ListTool(root).run()
            self.assertEqual(len(facts.details["files"]), 200)
            self.assertEqual(facts.details["count"], 200)
            self.assertTrue(facts.truncated)

    def test_does_not_leak_nabd_runtime_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "main.py").write_text("print(1)", encoding="utf-8")
            (root / ".nabd").mkdir()
            (root / ".nabd" / "evidence.json").write_text("{}", encoding="utf-8")
            (root / ".nabd" / "backups").mkdir()
            (root / ".nabd" / "backups" / "main.py.backup.1").write_text("x", encoding="utf-8")
            facts = ListTool(root).run()
            files = facts.details["files"]
            self.assertEqual(files, ["src/main.py"])
            for path in files:
                self.assertNotIn(".nabd", Path(path).parts)


if __name__ == "__main__":
    unittest.main()
