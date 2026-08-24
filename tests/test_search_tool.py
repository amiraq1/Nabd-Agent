import tempfile
import unittest
from pathlib import Path

from nabd.raw_facts import RawFacts
from nabd.search_tool import SearchTool


class SearchToolTests(unittest.TestCase):
    def test_searches_text_and_ignores_env(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "main.py").write_text("needle = True", encoding="utf-8")
            (root / "notes.txt").write_text("needle", encoding="utf-8")
            (root / ".env").write_text("needle=secret", encoding="utf-8")
            facts = SearchTool(root).run("needle")
            self.assertIsInstance(facts, RawFacts)
            self.assertEqual(facts.details["matches"], ["notes.txt", "src/main.py"])
            self.assertEqual(facts.details["result"], "MATCH")
            self.assertIn("backend", facts.details)

    def test_search_limits_results_to_50(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(75):
                (root / f"file_{index:03d}.txt").write_text("needle", encoding="utf-8")
            facts = SearchTool(root).run("needle")
            self.assertEqual(len(facts.details["matches"]), 50)

    def test_no_match_is_distinct_from_tool_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            facts = SearchTool(root).run("missing")
            self.assertEqual(facts.status, "OK")
            self.assertEqual(facts.details["result"], "NO_MATCH")
            self.assertIsNone(facts.error)

    def test_empty_query_is_structured_tool_error(self):
        with tempfile.TemporaryDirectory() as directory:
            facts = SearchTool(Path(directory)).run("   ")
            self.assertEqual(facts.status, "TOOL_ERROR")
            self.assertIsNotNone(facts.error)

    def test_does_not_leak_dot_nabd_evidence_or_backups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "main.py").write_text("needle", encoding="utf-8")
            (root / ".nabd").mkdir()
            (root / ".nabd" / "evidence.json").write_text("needle", encoding="utf-8")
            (root / ".nabd" / "backups").mkdir()
            (root / ".nabd" / "backups" / "app.py.backup.123").write_text("needle", encoding="utf-8")
            facts = SearchTool(root).run("needle")
            matches = facts.details["matches"]
            self.assertEqual(matches, ["src/main.py"])
            for path in matches:
                self.assertNotIn(".nabd", Path(path).parts)


if __name__ == "__main__":
    unittest.main()
