import tempfile
import unittest
from pathlib import Path

from nabd.agent import NabdAgent, _as_plan


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
    def test_plan_sanitizes_tool_name_from_verification(self):
        plan = _as_plan({"verification": ["run_command", "run_command: python hello.py", "run_command python3 -m compileall .", "python3 -m compileall ."]})
        self.assertEqual(plan.verification, ["python hello.py", "python3 -m compileall .", "python3 -m compileall ."])

    def test_agent_reaches_completed_after_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = NabdAgent(Path(directory), auto_approve=True)
            agent.client = FakeClient()
            result = agent.run("أنشئ ملف اختبار", max_rounds=1)
            self.assertTrue(result.ok)
            self.assertEqual(result.state, "COMPLETED")
            self.assertTrue((Path(directory) / "hello.py").exists())

    def test_verification_strips_hallucinated_tool_names(self):
        plan = _as_plan(
            {
                "verification": [
                    "run_command",                   # bare hallucination -> dropped
                    "run_command: python hello.py",  # colon form -> keeps command
                    "run_command flake8 .",          # space form (real bug) -> keeps command
                    "python3 -m compileall .",       # genuine command -> kept
                    "write_file",                    # bare hallucination -> dropped
                ]
            }
        )
        self.assertEqual(
            plan.verification,
            ["python hello.py", "flake8 .", "python3 -m compileall ."],
        )


if __name__ == "__main__":
    unittest.main()
