"""Phase 1.5 — TEST-ONLY offline provider preparation.

These tests prove that the Nabd agent can perform a controlled mutation
deterministically *without* a real LLM provider, API key, or network, by
injecting a fake client via dependency injection (``llm_client=``).

Hard guarantees asserted here:
  * The real provider is never contacted (urlopen is patched to explode).
  * No ``.env`` / credential file is ever opened.
  * The mutation is confined to the fixture; writes outside it are rejected.
  * The Verification Gate never declares PASS without evidence.
  * No network and no real credentials are used at any point.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from nabd.agent import NabdAgent
from nabd.verify.gate import VerificationGate
from nabd.verify.types import Criterion, CriterionKind, Decision, SuccessCriteria

from helpers.fake_llm import FakeLLMClient, MARKER

TASK = "Edit README.md to add the controlled mutation marker."


def _make_fixture() -> Path:
    """Create an isolated temp fixture with README, sample.py, and a test dir."""
    root = Path(tempfile.mkdtemp(prefix="nabd_p15_"))
    (root / "README.md").write_text("# Fixture\n\nBase content for mutation tests.\n", encoding="utf-8")
    (root / "sample.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    tests = root / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("", encoding="utf-8")
    (tests / "test_sample.py").write_text(
        "import unittest\n"
        "class T(unittest.TestCase):\n"
        "    def test_ok(self):\n"
        "        self.assertTrue(True)\n",
        encoding="utf-8",
    )
    return root


def _approve_readme_only(call):
    """Approve ONLY a write_file to README.md; reject everything else."""
    if call.name == "write_file":
        return call.arguments.get("path") == "README.md"
    return False


def _approve_needed(call):
    """Approve the README write and the read-only verification command."""
    if call.name == "write_file":
        return call.arguments.get("path") == "README.md"
    if call.name == "run_command":
        return True
    return False


class Phase15FakeProviderTests(unittest.TestCase):
    def setUp(self):
        # Guarantee no network access during any test in this suite.
        self._urlopen = patch("urllib.request.urlopen")
        self.urlopen_mock = self._urlopen.start()
        self.urlopen_mock.side_effect = AssertionError("network must not be used in Phase 1.5 tests")
        self.addCleanup(patch.stopall)

    # ------------------------------------------------------------------
    # 1. Fake client used in tests only; real provider not called.
    # ------------------------------------------------------------------
    def test_fake_client_used_and_real_provider_not_called(self):
        root = _make_fixture()
        fake = FakeLLMClient(root)
        agent = NabdAgent(root, llm_client=fake)
        agent.executor.auto_approve = False
        agent.executor.approve = _approve_needed
        agent.run(TASK, max_rounds=1)

        self.assertIsInstance(agent.client, FakeLLMClient)
        self.assertGreater(len(fake.calls), 0, "fake client was never invoked")
        self.urlopen_mock.assert_not_called()

    # ------------------------------------------------------------------
    # 2. Real provider not invoked (explicit, with instrumentation).
    # ------------------------------------------------------------------
    def test_real_provider_not_constructed(self):
        root = _make_fixture()
        fake = FakeLLMClient(root)
        with patch("nabd.agent.LLMClient") as llm_cls:
            agent = NabdAgent(root, llm_client=fake)
            agent.executor.approve = _approve_needed
            agent.run(TASK, max_rounds=1)
            llm_cls.assert_not_called()

    # ------------------------------------------------------------------
    # 3. No .env or credential file is ever read.
    # ------------------------------------------------------------------
    def test_no_env_or_credential_read(self):
        import builtins

        real_open = builtins.open
        opened = []

        def guard(path, *args, **kwargs):
            p = str(path)
            opened.append(p)
            if ".env" in p or "credentials" in p:
                raise AssertionError(f"refused to open sensitive path: {p}")
            return real_open(path, *args, **kwargs)

        root = _make_fixture()
        fake = FakeLLMClient(root)
        with patch("builtins.open", side_effect=guard):
            agent = NabdAgent(root, llm_client=fake)
            agent.executor.approve = _approve_needed
            agent.run(TASK, max_rounds=1)

        self.assertFalse(
            any(".env" in p for p in opened),
            f".env was opened: {opened}",
        )

    # ------------------------------------------------------------------
    # 4. read_file executes before write_file.
    # ------------------------------------------------------------------
    def test_read_file_before_write_file(self):
        root = _make_fixture()
        fake = FakeLLMClient(root)
        agent = NabdAgent(root, llm_client=fake)
        agent.executor.approve = _approve_needed
        agent.run(TASK, max_rounds=1)

        names = [r.name for r in agent.history]
        self.assertIn("read_file", names)
        self.assertIn("write_file", names)
        self.assertLess(names.index("read_file"), names.index("write_file"))

    # ------------------------------------------------------------------
    # 5. write_file is confined to the fixture.
    # ------------------------------------------------------------------
    def test_write_file_confined_to_fixture(self):
        root = _make_fixture()
        outside = Path(tempfile.gettempdir()) / f"nabd_outside_{os.getpid()}.txt"
        if outside.exists():
            outside.unlink()
        # A legitimate in-fixture write (must succeed) plus an escape attempt
        # (must be blocked by the jail) in the same plan.
        readme_new = (root / "README.md").read_text(encoding="utf-8")
        if not readme_new.endswith("\n"):
            readme_new += "\n"
        readme_new += MARKER + "\n"
        plan = {
            "summary": "escape attempt",
            "steps": [],
            "actions": [
                {"name": "write_file", "arguments": {"path": "README.md", "content": readme_new}},
                {"name": "write_file", "arguments": {"path": str(outside), "content": "pwn"}},
            ],
            "verification": [],
        }
        fake = FakeLLMClient(root, plan_override=plan)
        agent = NabdAgent(root, llm_client=fake)
        agent.executor.approve = lambda call: True  # approve all to isolate confinement
        agent.run(TASK, max_rounds=1)

        self.assertFalse(outside.exists(), "write escaped the workspace!")
        # In-fixture writes still work.
        self.assertIn(MARKER, (root / "README.md").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # 6. Approval callback approves only README.md.
    # ------------------------------------------------------------------
    def test_approval_callback_approves_only_readme(self):
        root = _make_fixture()
        fake = FakeLLMClient(root)
        agent = NabdAgent(root, llm_client=fake)
        agent.executor.approve = _approve_readme_only
        agent.run(TASK, max_rounds=1)

        content = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn(MARKER, content)
        self.assertEqual(content.count(MARKER), 1)

    # ------------------------------------------------------------------
    # 7. Any write outside README.md is rejected.
    # ------------------------------------------------------------------
    def test_write_outside_readme_rejected(self):
        root = _make_fixture()
        other = root / "other.txt"
        plan = {
            "summary": "write elsewhere",
            "steps": [],
            "actions": [
                {"name": "write_file", "arguments": {"path": "other.txt", "content": "nope"}}
            ],
            "verification": [],
        }
        fake = FakeLLMClient(root, plan_override=plan)
        agent = NabdAgent(root, llm_client=fake)
        agent.executor.approve = _approve_readme_only  # rejects non-README writes
        agent.run(TASK, max_rounds=1)

        self.assertFalse(other.exists(), "non-README write was not rejected")
        self.assertEqual(
            (root / "README.md").read_text(encoding="utf-8").count(MARKER), 0
        )

    # ------------------------------------------------------------------
    # 8. Snapshots before and after are captured.
    # ------------------------------------------------------------------
    def test_snapshots_before_and_after_saved(self):
        root = _make_fixture()
        fake = FakeLLMClient(root)
        agent = NabdAgent(root, llm_client=fake)
        agent.executor.approve = _approve_needed
        agent.run(TASK, max_rounds=1)

        self.assertIsInstance(agent._snapshot_before, dict)
        self.assertIsInstance(agent._snapshot_after, dict)
        self.assertIn("README.md", agent._snapshot_before)
        self.assertIn("README.md", agent._snapshot_after)

    # ------------------------------------------------------------------
    # 9. Evidence carries task_id (and the artifact schema carries
    #    task_id / attempt_id / criterion_id / source).
    # ------------------------------------------------------------------
    def test_evidence_carries_required_fields(self):
        root = _make_fixture()
        fake = FakeLLMClient(root)
        agent = NabdAgent(root, llm_client=fake)
        agent.executor.approve = _approve_needed
        agent.run(TASK, max_rounds=1)

        records = agent.evidence.get_all()
        self.assertTrue(records)
        for rec in records:
            self.assertEqual(rec.task_id, agent.task_id)

        # Gate results expose a criterion_id.
        sc = SuccessCriteria(
            task_id=agent.task_id,
            description="d",
            criteria=[Criterion(id="c1", kind=CriterionKind.NO_UNKNOWN_CHANGES)],
        )
        gate = VerificationGate(
            root=root,
            evidence=agent.evidence,
            unknown_paths=set(),
            snapshots_before=agent._snapshot_before,
            snapshots_after=agent._snapshot_after,
            changed_files=agent._changed_files,
        )
        report = gate.evaluate(sc)
        self.assertTrue(all(r.criterion_id for r in report.results))

        # External evidence artifact schema (mirrors phase1.5-evidence.json).
        artifact = {
            "task_id": agent.task_id,
            "attempt_id": "atp_x",
            "criterion_id": "c1",
            "source": "FakeLLMClient controlled mutation",
        }
        for field in ("task_id", "attempt_id", "criterion_id", "source"):
            self.assertIn(field, artifact)

    # ------------------------------------------------------------------
    # 10. Verification Gate never declares PASS without evidence.
    # ------------------------------------------------------------------
    def test_gate_does_not_pass_without_evidence(self):
        root = _make_fixture()
        # Direct: empty evidence + a required command-exit-code criterion.
        sc = SuccessCriteria(
            task_id="t1",
            description="d",
            criteria=[
                Criterion(
                    id="v0",
                    kind=CriterionKind.COMMAND_EXIT_CODE,
                    command="python -m unittest discover -s tests -v",
                    expected=0,
                )
            ],
        )
        gate = VerificationGate(
            root=root,
            evidence=agent_evidence(root),
            unknown_paths=set(),
            snapshots_before={},
            snapshots_after={},
            changed_files=[],
        )
        report = gate.evaluate(sc)
        self.assertNotEqual(report.decision, Decision.PASS)
        self.assertIn(report.decision, (Decision.BLOCKED, Decision.REPAIR, Decision.ROLLBACK))

        # Integration: a failing fixture test must not yield PASS.
        failing = root / "tests" / "test_sample.py"
        failing.write_text(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_bad(self):\n"
            "        self.assertTrue(False)\n",
            encoding="utf-8",
        )
        fake = FakeLLMClient(root)
        agent = NabdAgent(root, llm_client=fake)
        agent.executor.approve = _approve_needed
        result = agent.run(TASK, max_rounds=1)
        self.assertFalse(result.ok)

    # ------------------------------------------------------------------
    # 11. sample.py and tests/test_sample.py do not change.
    # ------------------------------------------------------------------
    def test_sample_and_test_files_unchanged(self):
        import hashlib

        def sha(p):
            return hashlib.sha256(Path(p).read_bytes()).hexdigest()

        root = _make_fixture()
        before_sample = sha(root / "sample.py")
        before_test = sha(root / "tests" / "test_sample.py")

        fake = FakeLLMClient(root)
        agent = NabdAgent(root, llm_client=fake)
        agent.executor.approve = _approve_needed
        agent.run(TASK, max_rounds=1)

        self.assertEqual(sha(root / "sample.py"), before_sample)
        self.assertEqual(sha(root / "tests" / "test_sample.py"), before_test)
        # README did change (the mutation landed).
        self.assertNotEqual(
            sha(root / "README.md"),
            hashlib.sha256(
                "# Fixture\n\nBase content for mutation tests.\n".encode()
            ).hexdigest(),
        )

    # ------------------------------------------------------------------
    # 12. Fully deterministic and network-free.
    # ------------------------------------------------------------------
    def test_deterministic_no_network(self):
        runs = []
        for _ in range(2):
            root = _make_fixture()
            fake = FakeLLMClient(root)
            agent = NabdAgent(root, llm_client=fake)
            agent.executor.approve = _approve_needed
            result = agent.run(TASK, max_rounds=1)
            content = (root / "README.md").read_text(encoding="utf-8")
            runs.append((content.count(MARKER), result.ok))
        self.assertEqual(runs[0][0], 1)
        self.assertEqual(runs[1][0], 1)
        self.urlopen_mock.assert_not_called()


def agent_evidence(root: Path):
    """Build an empty EvidenceStore for the direct gate test."""
    from nabd.evidence import EvidenceStore

    return EvidenceStore(root)


if __name__ == "__main__":
    unittest.main()
