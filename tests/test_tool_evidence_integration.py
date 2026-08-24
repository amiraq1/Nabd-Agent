import tempfile
import unittest
from pathlib import Path

from nabd.evidence import EvidenceStore, EvidenceType
from nabd.list_tool import ListTool
from nabd.read_tool import ReadTool
from nabd.raw_facts import RawFacts
from nabd.search_tool import SearchTool
from nabd.shell_tool import ShellTool
from nabd.tools import ToolExecutor
from nabd.models import ToolCall
from nabd.write_tool import WriteTool


class ToolEvidenceIntegrationTests(unittest.TestCase):
    def test_all_standalone_tools_return_raw_facts_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_id = "task-tools"
            store = EvidenceStore(root, task_id=task_id)
            write_facts = WriteTool(root).run("hello.py", "print('hello')")
            read_facts = ReadTool(root).run("hello.py")
            list_facts = ListTool(root).run()
            search_facts = SearchTool(root).run("hello")
            shell_facts = ShellTool(root).run("python3 -m py_compile hello.py")
            for facts in (write_facts, read_facts, list_facts, search_facts, shell_facts):
                self.assertIsInstance(facts, RawFacts)
            self.assertEqual(store.get_all(), [])
            evidence = [store.verify(facts, task_id=task_id) for facts in (write_facts, read_facts, list_facts, search_facts, shell_facts)]
            self.assertTrue(all(item.evidence_type is EvidenceType.OBSERVED for item in evidence))
            self.assertTrue(store.is_usable_for_completion(task_id))

    def test_old_task_evidence_cannot_prove_current_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            facts = WriteTool(root).run("hello.txt", "hello")
            store = EvidenceStore(root, task_id="task-current")
            evidence = store.verify(facts, task_id="task-old")
            self.assertEqual(evidence.evidence_type, EvidenceType.INFERRED)
            self.assertFalse(store.is_usable_for_completion("task-current"))

    def test_tampered_raw_fact_is_not_observed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            facts = WriteTool(root).run("hello.txt", "hello")
            (root / "hello.txt").write_text("tampered", encoding="utf-8")
            store = EvidenceStore(root, task_id="task-1")
            evidence = store.verify(facts, task_id="task-1")
            self.assertEqual(evidence.evidence_type, EvidenceType.INFERRED)
            self.assertFalse(store.is_usable_for_completion("task-1"))

    def test_tool_executor_exposes_search_as_raw_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("needle = True", encoding="utf-8")
            executor = ToolExecutor(root, auto_approve=True)
            result = executor.execute(ToolCall("search", {"query": "needle"}))
            self.assertTrue(result.ok)
            self.assertIsInstance(result.raw_facts, RawFacts)
            self.assertIn("app.py", result.output)

    def test_failed_shell_is_not_completion_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            facts = ShellTool(root).run("python3 -c 'raise SystemExit(3)'")
            store = EvidenceStore(root, task_id="task-1")
            evidence = store.verify(facts, task_id="task-1")
            self.assertEqual(evidence.evidence_type, EvidenceType.INFERRED)
            self.assertFalse(store.is_usable_for_completion("task-1"))


if __name__ == "__main__":
    unittest.main()
