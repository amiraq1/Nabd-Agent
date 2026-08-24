import tempfile
import unittest
from pathlib import Path

from nabd.agent import NabdAgent


class FakeClient:
    provider = "fake"
    model = "test"

    def complete_json(self, _system, _user):
        return {
            "summary": "إنشاء ملف Python بسيط والتحقق من صياغته",
            "steps": ["إنشاء hello.py", "فحص الصياغة"],
            "actions": [
                {
                    "name": "write_file",
                    "arguments": {"path": "hello.py", "content": "print('hello')\n"},
                }
            ],
            "verification": ["python3 -m py_compile hello.py"],
        }


class AgentTests(unittest.TestCase):
    def test_agent_reaches_completed_after_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = NabdAgent(Path(directory), auto_approve=True)
            agent.client = FakeClient()
            result = agent.run("أنشئ ملف اختبار", max_rounds=1)
            self.assertTrue(result.ok)
            self.assertEqual(result.state, "COMPLETED")
            self.assertTrue((Path(directory) / "hello.py").exists())


if __name__ == "__main__":
    unittest.main()
