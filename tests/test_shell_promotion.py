import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Allow NabdAgent construction without a real LLM provider for offline tests.
os.environ.setdefault("NVIDIA_API_KEY", "dummy-for-offline-tests")

from nabd.agent import NabdAgent, classify_intent
from nabd.models import ToolCall
from nabd.tools import ToolExecutor, is_plausible_shell_command


ARABIC_PROSE = "استخدم tool list_files لاستعراض بنية مساحة الاختبار"
ENGLISH_PROSE = "use the list_files tool to inspect the project structure"


class PlausibleShellCommandTests(unittest.TestCase):
    def test_valid_commands_are_plausible(self):
        for cmd in [
            "pwd",
            "ls -la",
            "git status --short",
            "python3 -m py_compile x.py",
            "grep -q foo bar",
            "cd dir && make",
            "sudo rm -rf x",
            "echo hi",
            "(cd x && make)",
            "bash -c 'echo hi'",
            "env PY=1 python3 -c 'print(1)'",
        ]:
            self.assertTrue(is_plausible_shell_command(cmd), cmd)

    def test_prose_is_not_plausible(self):
        for cmd in [
            ARABIC_PROSE,
            ENGLISH_PROSE,
            "please inspect the repository",
            "تحقق من بنية المشروع",
            "run the inspection tool now",
        ]:
            self.assertFalse(is_plausible_shell_command(cmd), cmd)


class ShellPromotionTests(unittest.TestCase):
    def test_mutating_prose_is_rejected_without_shell_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = ToolExecutor(root, auto_approve=True)
            executor.set_intent("MUTATING")
            executor.shell.run = MagicMock(side_effect=AssertionError("shell.run must not be called"))
            result = executor.execute(ToolCall("run_command", {"command": ARABIC_PROSE}))
            self.assertFalse(result.ok)
            self.assertEqual(result.raw_facts.status, "NOT_A_COMMAND")
            executor.shell.run.assert_not_called()

    def test_english_prose_is_rejected_without_shell_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = ToolExecutor(root, auto_approve=True)
            executor.set_intent("MUTATING")
            executor.shell.run = MagicMock(side_effect=AssertionError("shell.run must not be called"))
            result = executor.execute(ToolCall("run_command", {"command": ENGLISH_PROSE}))
            self.assertFalse(result.ok)
            self.assertEqual(result.raw_facts.status, "NOT_A_COMMAND")
            executor.shell.run.assert_not_called()

    def test_read_only_prose_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = ToolExecutor(root, auto_approve=True)
            executor.set_intent("READ_ONLY")
            executor.shell.run = MagicMock(side_effect=AssertionError("shell.run must not be called"))
            result = executor.execute(ToolCall("run_command", {"command": ARABIC_PROSE}))
            self.assertFalse(result.ok)
            self.assertIn(result.raw_facts.status, {"NOT_A_COMMAND", "MUTATION_NOT_ALLOWED"})
            executor.shell.run.assert_not_called()

    def test_legitimate_mutating_command_still_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = ToolExecutor(root, auto_approve=True)
            executor.set_intent("MUTATING")
            target = root / "created.txt"
            result = executor.execute(
                ToolCall("run_command", {"command": f'touch "{target}"'})
            )
            self.assertTrue(result.ok, result.output)
            self.assertTrue(target.exists())


class IntentNegationTests(unittest.TestCase):
    def test_negated_mutating_keyword_is_read_only(self):
        self.assertEqual(classify_intent("افحص بنية المشروع فقط، لا تعدّل أي ملف"), "READ_ONLY")
        self.assertEqual(classify_intent("don't edit any file"), "READ_ONLY")
        self.assertEqual(classify_intent("do not delete the logs"), "READ_ONLY")

    def test_non_negated_mutating_keyword_still_mutating(self):
        self.assertEqual(classify_intent("افحص ثم أصلح الخطأ"), "MUTATING")
        self.assertEqual(classify_intent("inspect and fix the failing test"), "MUTATING")

    def test_read_only_keyword_still_read_only(self):
        self.assertEqual(classify_intent("افحص المستودع واعطني ملخص"), "READ_ONLY")


class AgentProseIntegrationTests(unittest.TestCase):
    def test_agent_never_executes_prose_verification(self):
        sentinel = Path(tempfile.gettempdir()) / "nabd_prose_sentinel.txt"

        class ProseClient:
            provider = "fake"
            model = "test"

            def complete_json(self, _system, _user):
                return {
                    "summary": "inspect the project",
                    "steps": ["inspect"],
                    "actions": [{"name": "list_files", "arguments": {"path": "."}}],
                    "verification": [ARABIC_PROSE],
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = NabdAgent(root, auto_approve=True, workspace_free=True)
            agent.client = ProseClient()

            def _shell_run(*_a, **_k):
                sentinel.write_text("executed")
                raise AssertionError("shell.run was called with prose")

            agent.executor.shell.run = _shell_run
            if sentinel.exists():
                sentinel.unlink()
            result = agent.run("inspect the project structure", max_rounds=2)
            self.assertFalse(sentinel.exists())
            self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main()
