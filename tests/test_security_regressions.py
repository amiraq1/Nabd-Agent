import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from nabd.agent import NabdAgent
from nabd.llm import LLMClient, LLMError
from nabd.models import ToolCall
from nabd.snapshot import SnapshotError, SnapshotStore
from nabd.tools import ToolExecutor


class _Response:
    def __init__(self, payload: dict):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class SecurityRegressionTests(unittest.TestCase):
    def test_approval_allowlist_denies_negative_and_unknown_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = NabdAgent(Path(directory), provider="openai")
            call = ToolCall("write_file", {"path": "blocked.txt", "content": "blocked"})
            for answer in ("n", "no", "لا", "ن", "", "maybe", "unknown"):
                with self.subTest(answer=answer), patch("builtins.input", return_value=answer):
                    self.assertFalse(agent._approve(call))
            for answer in ("y", "Y", "yes", "YES", "نعم"):
                with self.subTest(answer=answer), patch("builtins.input", return_value=answer):
                    self.assertTrue(agent._approve(call))
            with patch("builtins.input", side_effect=EOFError):
                self.assertFalse(agent._approve(call))
            with patch("builtins.input", side_effect=KeyboardInterrupt):
                self.assertFalse(agent._approve(call))

    def test_denied_approval_never_dispatches_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = ToolExecutor(root, approve=lambda _call: False, auto_approve=False)
            executor._dispatch = Mock(side_effect=AssertionError("dispatch must not run"))
            result = executor.execute(
                ToolCall("write_file", {"path": "blocked.txt", "content": "blocked"})
            )
            self.assertFalse(result.ok)
            self.assertEqual(result.exit_code, 126)
            executor._dispatch.assert_not_called()
            self.assertFalse((root / "blocked.txt").exists())

    def test_snapshot_failure_rejects_before_inventory_planning_or_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = NabdAgent(Path(directory), provider="openai")
            agent.client = Mock()
            agent.executor.execute = Mock(side_effect=AssertionError("tool must not run"))
            with patch.object(
                agent.snapshot,
                "load_or_create",
                side_effect=SnapshotError("simulated snapshot write failure"),
            ):
                result = agent.run("أنشئ ملفًا", max_rounds=1)
            self.assertFalse(result.ok)
            self.assertEqual(result.state, "REJECTED")
            agent.client.complete_json.assert_not_called()
            agent.executor.execute.assert_not_called()
            self.assertEqual(len(result.evidence), 1)
            self.assertEqual(result.evidence[0]["type"], "INFERRED")
            self.assertTrue((Path(directory) / ".nabd" / "evidence.json").is_file())

    def test_corrupt_persisted_snapshot_is_rebuilt_before_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SnapshotStore(root)
            path = root / ".nabd" / "snapshot-before.json"
            path.parent.mkdir()
            path.write_text("{not-json", encoding="utf-8")
            manifest = store.load_or_create("task-test")
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["task_id"], "task-test")
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["task_id"], "task-test")

    def test_gemini_key_is_header_only_and_not_in_url(self):
        secret = "gemini-regression-secret"
        response = {
            "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]
        }
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _Response(response)

        with patch.dict("os.environ", {"GEMINI_API_KEY": secret}, clear=False):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                result = LLMClient("gemini").complete_json("system", "user")

        request = captured["request"]
        self.assertEqual(result, {"ok": True})
        self.assertNotIn(secret, request.full_url)
        headers = {key.lower(): value for key, value in request.header_items()}
        self.assertEqual(headers["x-goog-api-key"], secret)

    def test_gemini_unexpected_response_redacts_key(self):
        secret = "gemini-response-secret"

        def fake_urlopen(_request, **_kwargs):
            return _Response({"error": "secret=" + secret})

        with patch.dict("os.environ", {"GEMINI_API_KEY": secret}, clear=False):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                with self.assertRaises(LLMError) as caught:
                    LLMClient("gemini").complete_json("system", "user")
        self.assertNotIn(secret, str(caught.exception))
        self.assertIn("[REDACTED]", str(caught.exception))

    def test_gemini_http_error_redacts_key(self):
        secret = "gemini-error-secret"

        def fake_urlopen(_request, **_kwargs):
            error = __import__("urllib.error", fromlist=["HTTPError"]).HTTPError(
                "https://generativelanguage.googleapis.com/v1beta/models/test:generateContent",
                403,
                "forbidden",
                {},
                io.BytesIO(("secret=" + secret).encode("utf-8")),
            )
            raise error

        with patch.dict("os.environ", {"GEMINI_API_KEY": secret}, clear=False):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                with self.assertRaises(LLMError) as caught:
                    LLMClient("gemini").complete_json("system", "user")
        self.assertNotIn(secret, str(caught.exception))
        self.assertIn("[REDACTED]", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
