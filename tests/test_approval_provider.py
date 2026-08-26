import inspect
import tempfile
import unittest
from pathlib import Path

from nabd.agent import NabdAgent
from nabd.approval import ApprovalMode, CallbackApprovalProvider
from nabd.models import ToolCall
from nabd.runtime import run_task


class ApprovalProviderTests(unittest.TestCase):
    def test_runtime_requires_keyword_only_approval_mode(self):
        parameter = inspect.signature(run_task).parameters["approval_mode"]
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(parameter.default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            run_task(root=tempfile.mkdtemp(), provider="openai")

    def test_invalid_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                NabdAgent(Path(directory), provider="openai", approval_mode="MAYBE")

    def test_callback_failure_denies(self):
        provider = CallbackApprovalProvider(lambda _request: (_ for _ in ()).throw(RuntimeError("UI failed")))
        self.assertFalse(provider.decide({"tool": "write_file"}))

    def test_auto_requires_explicit_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = NabdAgent(Path(directory), provider="openai", approval_mode=ApprovalMode.AUTO)
            self.assertTrue(agent._approve(ToolCall("write_file", {"path": "x", "content": "x"})))

    def test_legacy_constructor_auto_flag_is_compatible_but_runtime_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = NabdAgent(Path(directory), provider="openai", auto_approve=True)
            self.assertEqual(agent.approval_mode, ApprovalMode.AUTO)

    def test_workspace_free_cannot_be_confirm_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                NabdAgent(
                    Path(directory),
                    provider="openai",
                    workspace_free=True,
                    approval_mode=ApprovalMode.CONFIRM,
                )


if __name__ == "__main__":
    unittest.main()
