import tempfile
import unittest
from pathlib import Path

from nabd.agent import NabdAgent, classify_intent
from nabd.models import ToolCall
from nabd.raw_facts import RawFacts
from nabd.tools import ToolExecutor, is_read_only_shell_command


class ReadOnlyWriteClient:
    provider = "fake"
    model = "test"

    def complete_json(self, _system, _user):
        return {
            "summary": "فحص المستودع وتقديم ملخص",
            "steps": ["إنشاء README للملخص"],
            "actions": [
                {
                    "name": "write_file",
                    "arguments": {"path": "README.md", "content": "must not be written"},
                }
            ],
            "verification": ["python3 -c 'print(\"unrelated success\")'"],
        }


class MutationPolicyTests(unittest.TestCase):
    def test_read_only_keywords_are_classified_read_only(self):
        self.assertEqual(classify_intent("افحص المستودع واعطني ملخص"), "READ_ONLY")
        self.assertEqual(classify_intent("inspect the repository and report"), "READ_ONLY")

    def test_mutating_keyword_wins_over_read_only_keyword(self):
        self.assertEqual(classify_intent("افحص ثم أصلح الخطأ"), "MUTATING")
        self.assertEqual(classify_intent("inspect and fix the failing test"), "MUTATING")

    def test_unknown_intent_defaults_to_mutating(self):
        self.assertEqual(classify_intent("نفذ المهمة المطلوبة"), "MUTATING")

    def test_read_only_write_is_structured_policy_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = ToolExecutor(root, auto_approve=True)
            executor.set_intent("READ_ONLY")

            result = executor.execute(
                ToolCall(
                    "write_file",
                    {"path": "README.md", "content": "must not be written"},
                )
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.exit_code, 126)
            self.assertIn("MUTATION_NOT_ALLOWED", result.output)
            self.assertIsInstance(result.raw_facts, RawFacts)
            self.assertEqual(result.raw_facts.status, "MUTATION_NOT_ALLOWED")
            self.assertEqual(result.raw_facts.details["intent"], "READ_ONLY")
            self.assertFalse((root / "README.md").exists())

    def test_read_only_shell_classifier_is_conservative(self):
        self.assertTrue(is_read_only_shell_command("pwd"))
        self.assertTrue(is_read_only_shell_command("git status --short --no-optional-locks"))
        self.assertFalse(is_read_only_shell_command("git status --short"))
        self.assertFalse(is_read_only_shell_command("echo x > output.txt"))
        self.assertFalse(is_read_only_shell_command("python3 -c 'open(\"x\", \"w\").write(\"x\")'"))
        self.assertFalse(is_read_only_shell_command("find . -delete"))

    def test_read_only_shell_mutation_is_blocked_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = ToolExecutor(root, auto_approve=True)
            executor.set_intent("READ_ONLY")

            result = executor.execute(
                ToolCall(
                    "run_command",
                    {"command": "python3 -c 'open(\"created.txt\", \"w\").write(\"bad\")'"},
                )
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.exit_code, 126)
            self.assertIn("MUTATION_NOT_ALLOWED", result.output)
            self.assertIsInstance(result.raw_facts, RawFacts)
            self.assertEqual(result.raw_facts.operation, "shell")
            self.assertFalse((root / "created.txt").exists())

    def test_read_only_inspection_shell_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = ToolExecutor(root, auto_approve=True)
            executor.set_intent("READ_ONLY")
            result = executor.execute(ToolCall("run_command", {"command": "pwd"}))
            self.assertTrue(result.ok)
            self.assertIn(str(root), result.output)

    def test_mutating_task_can_run_mutating_shell_when_auto_approved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = ToolExecutor(root, auto_approve=True)
            executor.set_intent("MUTATING")
            result = executor.execute(
                ToolCall(
                    "run_command",
                    {"command": "python3 -c 'open(\"created.txt\", \"w\").write(\"ok\")'"},
                )
            )
            self.assertTrue(result.ok)
            self.assertEqual((root / "created.txt").read_text(encoding="utf-8"), "ok")

    def test_mutating_task_can_write_when_auto_approved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = ToolExecutor(root, auto_approve=True)
            executor.set_intent("MUTATING")

            result = executor.execute(
                ToolCall(
                    "write_file",
                    {"path": "README.md", "content": "allowed"},
                )
            )

            self.assertTrue(result.ok)
            self.assertEqual((root / "README.md").read_text(encoding="utf-8"), "allowed")

    def test_agent_read_only_policy_blocks_write_before_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = NabdAgent(root, auto_approve=True)
            agent.client = ReadOnlyWriteClient()

            result = agent.run("افحص المستودع واعطني ملخص", max_rounds=1)

            self.assertFalse(result.ok)
            self.assertEqual(result.state, "REJECTED")
            self.assertFalse((root / "README.md").exists())
            self.assertTrue(any(item["details"].get("intent") == "READ_ONLY" for item in result.evidence))

    def test_invalid_intent_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = ToolExecutor(Path(directory), auto_approve=True)
            with self.assertRaises(ValueError):
                executor.set_intent("UNKNOWN")


if __name__ == "__main__":
    unittest.main()
