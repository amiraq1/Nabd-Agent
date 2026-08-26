import tempfile
import unittest
from pathlib import Path

from nabd.raw_facts import RawFacts
from nabd.shell_tool import ShellTool


class ShellToolTests(unittest.TestCase):
    def test_runs_inside_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            facts = ShellTool(Path(directory)).run("pwd")
            self.assertIsInstance(facts, RawFacts)
            self.assertEqual(facts.exit_code, 0)
            self.assertEqual(Path((facts.stdout or "").strip()), Path(directory).resolve())
            self.assertFalse(facts.details.get("timeout", False))

    def test_blocks_dangerous_command_as_tool_error(self):
        with tempfile.TemporaryDirectory() as directory:
            facts = ShellTool(Path(directory)).run("sudo rm -rf /")
            self.assertEqual(facts.status, "TOOL_ERROR")
            self.assertIsNotNone(facts.error)

    def test_returns_nonzero_command_status(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "exit3.py"
            script.write_text("import sys\nsys.exit(3)\n")
            facts = ShellTool(Path(directory)).run(f"python3 {script}")
            self.assertEqual(facts.exit_code, 3)
            self.assertFalse(facts.successful)


if __name__ == "__main__":
    unittest.main()
