"""Lifecycle integration tests for the Nabd agent.

Each scenario uses a deterministic fake LLM provider so tests are
reproducible without API keys.  A temporary sandbox workspace is
created per test with a buggy ``calculator.py`` and a verification
command that uses ``python3 -B -c`` (no pytest needed, no bytecode
caching).

Scenarios tested:
  1. PASS     – correct fix → COMPLETED
  2. REPAIR   – first fix broken, second correct → COMPLETED
  3. ROLLBACK – repeated identical failure → ROLLED_BACK
  4. BLOCKED  – external change detected → BLOCKED
  5. TIMEOUT  – wall-clock exceeded → FAILED
  6. Snapshot – diff utilities work correctly
  7. Gate     – gate evaluation with real filesystem
  8. FSM      – transition consistency
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from nabd.agent import NabdAgent
from nabd.evidence import EvidenceStore
from nabd.fsm import FSM, State
from nabd.models import AgentResult, Plan, ToolCall, ToolResult
from nabd.verify.gate import VerificationGate, diff_snapshots, take_snapshot
from nabd.verify.types import (
    Criterion,
    CriterionKind,
    Decision,
    FailureSignature,
    Report,
    Result,
    SuccessCriteria,
)


# ────────────────────────────────────────────────────────────────────────
# Sandbox helpers
# ────────────────────────────────────────────────────────────────────────

_BUGGY_CALCULATOR = textwrap.dedent("""\
    def add(a, b):
        return a - b

    def multiply(a, b):
        return a * b
""")

_FIXED_CALCULATOR = textwrap.dedent("""\
    def add(a, b):
        return a + b

    def multiply(a, b):
        return a * b
""")

_STILL_BROKEN_CALCULATOR = textwrap.dedent("""\
    def add(a, b):
        return a - b  # still broken!

    def multiply(a, b):
        return a * b
""")

_TEST_FILE = textwrap.dedent("""\
    from src.calculator import add, multiply


    def test_add():
        assert add(2, 3) == 5


    def test_multiply():
        assert multiply(2, 3) == 6
""")

# Use grep to check file content instead of Python imports to avoid
# .pyc caching issues.  Python reads existing .pyc even with -B, so
# a second import after overwrite may use stale bytecode.
_VERIFY_CMD = 'grep -q "return a + b" src/calculator.py'


def _setup_sandbox(root: Path) -> None:
    """Create the buggy calculator project."""
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "src" / "__init__.py").write_text("")
    (root / "src" / "calculator.py").write_text(_BUGGY_CALCULATOR)
    (root / "tests" / "__init__.py").write_text("")
    (root / "tests" / "test_calculator.py").write_text(_TEST_FILE)
    (root / "README.md").write_text("# Calculator\n")


# ────────────────────────────────────────────────────────────────────────
# Fake LLM providers
# ────────────────────────────────────────────────────────────────────────


class _FakeProvider:
    """Base fake provider with common helpers."""

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "test"


class _FakeProviderPass(_FakeProvider):
    """Returns a correct fix on the first call."""

    def complete_json(self, _system: str, _user: str) -> Dict[str, Any]:
        self.call_count += 1
        return {
            "summary": "Fix the add function in calculator.py",
            "steps": [
                "Read calculator.py to confirm the bug",
                "Fix add function: change a - b to a + b",
                "Run tests to verify the fix",
            ],
            "actions": [
                {"name": "read_file", "arguments": {"path": "src/calculator.py"}},
                {
                    "name": "write_file",
                    "arguments": {
                        "path": "src/calculator.py",
                        "content": _FIXED_CALCULATOR,
                    },
                },
            ],
            "verification": [_VERIFY_CMD],
        }


class _FakeProviderRepair(_FakeProvider):
    """First call: broken fix; second call: correct fix."""

    def complete_json(self, _system: str, _user: str) -> Dict[str, Any]:
        self.call_count += 1
        if self.call_count == 1:
            return {
                "summary": "Fix calculator",
                "steps": ["Fix add function partially"],
                "actions": [
                    {
                        "name": "write_file",
                        "arguments": {
                            "path": "src/calculator.py",
                            "content": _STILL_BROKEN_CALCULATOR,
                        },
                    },
                ],
                "verification": [_VERIFY_CMD],
            }
        return {
            "summary": "Fix calculator correctly",
            "steps": ["Fix add function properly"],
            "actions": [
                {
                    "name": "write_file",
                    "arguments": {
                        "path": "src/calculator.py",
                        "content": _FIXED_CALCULATOR,
                    },
                },
            ],
            "verification": [_VERIFY_CMD],
        }


class _FakeProviderRollback(_FakeProvider):
    """Always returns the same broken fix."""

    def complete_json(self, _system: str, _user: str) -> Dict[str, Any]:
        self.call_count += 1
        return {
            "summary": "Try to fix calculator",
            "steps": ["Apply fix"],
            "actions": [
                {
                    "name": "write_file",
                    "arguments": {
                        "path": "src/calculator.py",
                        "content": _STILL_BROKEN_CALCULATOR,
                    },
                },
            ],
            "verification": [_VERIFY_CMD],
        }


# ────────────────────────────────────────────────────────────────────────
# Scenario 1: PASS
# ────────────────────────────────────────────────────────────────────────


class LifecyclePassTests(unittest.TestCase):
    """Correct fix → COMPLETED."""

    def test_correct_fix_reaches_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _setup_sandbox(root)

            agent = NabdAgent(root, auto_approve=True)
            agent.client = _FakeProviderPass()

            result = agent.run("fix the calculator add function", max_rounds=3)

            self.assertTrue(result.ok, f"Agent failed: {result.error}")
            self.assertEqual(result.state, "COMPLETED")

            fixed_content = (root / "src" / "calculator.py").read_text()
            self.assertIn("return a + b", fixed_content)
            self.assertNotIn("return a - b", fixed_content)
            self.assertGreater(len(result.evidence), 0)

    def test_pass_records_verification_decision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _setup_sandbox(root)

            agent = NabdAgent(root, auto_approve=True)
            agent.client = _FakeProviderPass()

            result = agent.run("fix calculator", max_rounds=3)

            self.assertTrue(result.ok, f"Agent failed: {result.error}")
            decision_records = [
                ev for ev in result.evidence if ev.get("claim") == "verification_decision"
            ]
            self.assertGreater(len(decision_records), 0)
            details = decision_records[0].get("details", {})
            self.assertEqual(details.get("decision"), "PASS")


# ────────────────────────────────────────────────────────────────────────
# Scenario 2: REPAIR
# ────────────────────────────────────────────────────────────────────────


class LifecycleRepairTests(unittest.TestCase):
    """First fix fails, second succeeds → COMPLETED."""

    def test_repair_cycle_reaches_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _setup_sandbox(root)

            agent = NabdAgent(root, auto_approve=True)
            agent.client = _FakeProviderRepair()

            result = agent.run("fix calculator", max_rounds=5)

            self.assertTrue(result.ok, f"Agent failed: {result.error}")
            self.assertEqual(result.state, "COMPLETED")

            fixed_content = (root / "src" / "calculator.py").read_text()
            self.assertIn("return a + b", fixed_content)

    def test_repair_increments_repair_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _setup_sandbox(root)

            agent = NabdAgent(root, auto_approve=True)
            agent.client = _FakeProviderRepair()

            result = agent.run("fix calculator", max_rounds=5)

            self.assertGreaterEqual(agent._repair_count, 1)


# ────────────────────────────────────────────────────────────────────────
# Scenario 3: ROLLBACK
# ────────────────────────────────────────────────────────────────────────


class LifecycleRollbackTests(unittest.TestCase):
    """Same failure repeated → ROLLED_BACK."""

    def test_repeated_failure_triggers_rollback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _setup_sandbox(root)

            agent = NabdAgent(root, auto_approve=True)
            agent.client = _FakeProviderRollback()

            result = agent.run("fix calculator", max_rounds=5)

            self.assertFalse(result.ok)
            self.assertEqual(result.state, "ROLLED_BACK")

    def test_failure_signature_streak_detected(self):
        """Repeated identical failures must break the repair loop (no 3rd attempt)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _setup_sandbox(root)

            agent = NabdAgent(root, auto_approve=True)
            agent.client = _FakeProviderRollback()

            result = agent.run("fix calculator", max_rounds=5)

            # Must terminate (not loop forever) and end rolled back.
            self.assertFalse(result.ok)
            self.assertEqual(result.state, "ROLLED_BACK")
            # The same broken fix is returned every call; after two identical
            # failures the gate breaks the loop, so a 3rd repair is never tried.
            self.assertLessEqual(agent.client.call_count, 2)
            self.assertGreaterEqual(len(agent._failure_signatures), 1)

    def test_rollback_error_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _setup_sandbox(root)

            agent = NabdAgent(root, auto_approve=True)
            agent.client = _FakeProviderRollback()

            result = agent.run("fix calculator", max_rounds=5)

            self.assertFalse(result.ok)
            self.assertEqual(result.state, "ROLLED_BACK")
            self.assertIn("Rolled back", result.error or "")


# ────────────────────────────────────────────────────────────────────────
# Scenario 4: BLOCKED
# ────────────────────────────────────────────────────────────────────────


class LifecycleBlockedTests(unittest.TestCase):
    """External change detected → BLOCKED."""

    def test_external_change_blocks_completion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _setup_sandbox(root)

            agent = NabdAgent(root, auto_approve=True)
            agent.client = _FakeProviderPass()

            # Inject an unknown file before snapshot_after
            original_take_after = agent._take_snapshot_after

            def _take_with_external() -> None:
                (root / "EXTERNAL_CHANGE.md").write_text("external modification")
                original_take_after()

            agent._take_snapshot_after = _take_with_external

            result = agent.run("fix calculator", max_rounds=3)

            self.assertFalse(result.ok)
            self.assertIn("Blocked", result.error or "")


# ────────────────────────────────────────────────────────────────────────
# Scenario 5: TIMEOUT
# ────────────────────────────────────────────────────────────────────────


class LifecycleTimeoutTests(unittest.TestCase):
    """Wall-clock timeout exceeded → FAILED."""

    def test_timeout_triggers_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _setup_sandbox(root)

            agent = NabdAgent(root, auto_approve=True)
            agent.client = _FakeProviderPass()

            # Patch _check_timeout to always return True so the agent
            # sees a timeout at the start of the first round, before
            # it overwrites _start_time in run().
            with patch.object(agent, "_check_timeout", return_value=True):
                result = agent.run("fix calculator", max_rounds=3)

            self.assertFalse(result.ok)
            self.assertIn("Timeout", result.error or "")
            self.assertIn(agent.fsm.state, {State.FAILED, State.REJECTED})


# ────────────────────────────────────────────────────────────────────────
# Scenario 6: Dirty workspace
# ────────────────────────────────────────────────────────────────────────


class LifecycleDirtyWorkspaceTests(unittest.TestCase):
    """Manual changes before task → snapshot detects them."""

    def test_manual_change_detected_in_snapshot(self):
        """An external change injected AFTER snapshot_before but BEFORE
        snapshot_after should be detected as an unknown change."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _setup_sandbox(root)

            agent = NabdAgent(root, auto_approve=True)
            agent.client = _FakeProviderPass()

            # Inject external change AFTER the before-snapshot (which is
            # taken at the start of run()) but BEFORE the after-snapshot.
            original_take_after = agent._take_snapshot_after

            def _take_with_manual_change() -> None:
                # Simulate an external actor modifying README between
                # the before-snapshot and the after-snapshot.
                (root / "README.md").write_text("# Modified externally\n")
                original_take_after()

            agent._take_snapshot_after = _take_with_manual_change

            result = agent.run("fix calculator", max_rounds=3)

            # README.md changed externally → should be blocked
            self.assertFalse(result.ok)
            self.assertIn("Blocked", result.error or "")


# ────────────────────────────────────────────────────────────────────────
# Snapshot diff tests
# ────────────────────────────────────────────────────────────────────────


class SnapshotDiffTests(unittest.TestCase):
    """Snapshot diffing utilities."""

    def test_take_snapshot_covers_all_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _setup_sandbox(root)

            snap = take_snapshot(root)

            self.assertIn("src/calculator.py", snap)
            self.assertIn("tests/test_calculator.py", snap)
            self.assertIn("README.md", snap)
            for h in snap.values():
                self.assertEqual(len(h), 64)

    def test_diff_detects_added_file(self):
        before = {"a.txt": "hash_a", "b.txt": "hash_b"}
        after = {"a.txt": "hash_a", "b.txt": "hash_b", "c.txt": "hash_c"}
        diff = diff_snapshots(before, after)
        self.assertIn("c.txt", diff["added"])
        self.assertEqual(diff["removed"], [])
        self.assertEqual(diff["changed"], [])

    def test_diff_detects_changed_file(self):
        before = {"a.txt": "hash_a", "b.txt": "hash_b"}
        after = {"a.txt": "hash_a", "b.txt": "hash_b2"}
        diff = diff_snapshots(before, after)
        self.assertIn("b.txt", diff["changed"])

    def test_diff_detects_removed_file(self):
        before = {"a.txt": "hash_a", "b.txt": "hash_b"}
        after = {"a.txt": "hash_a"}
        diff = diff_snapshots(before, after)
        self.assertIn("b.txt", diff["removed"])

    def test_diff_no_changes(self):
        snap = {"a.txt": "hash_a", "b.txt": "hash_b"}
        diff = diff_snapshots(snap, snap)
        self.assertEqual(diff["added"], [])
        self.assertEqual(diff["removed"], [])
        self.assertEqual(diff["changed"], [])


# ────────────────────────────────────────────────────────────────────────
# Gate evaluation tests
# ────────────────────────────────────────────────────────────────────────


class GateEvaluationTests(unittest.TestCase):
    """Gate evaluation with real filesystem."""

    def test_gate_pass_with_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _setup_sandbox(root)
            (root / "src" / "calculator.py").write_text(_FIXED_CALCULATOR)

            evidence = EvidenceStore(root, task_id="test-task")

            evidence.add_observed_check(
                claim="verification: pytest",
                command="pytest",
                exit_code=0,
                output="2 passed",
                task_id="test-task",
            )

            criteria = SuccessCriteria(
                task_id="test-task",
                description="fix calculator",
                criteria=[
                    Criterion(id="tests_pass", kind=CriterionKind.COMMAND_EXIT_CODE, command="pytest", expected=0),
                    Criterion(id="no_unknown", kind=CriterionKind.NO_UNKNOWN_CHANGES),
                ],
            )

            gate = VerificationGate(root=root, evidence=evidence)
            report = gate.evaluate(criteria)

            self.assertEqual(report.decision, Decision.PASS)
            self.assertEqual(report.failed_required, [])

    def test_gate_blocks_with_unknown_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _setup_sandbox(root)

            evidence = EvidenceStore(root, task_id="test-task")
            evidence.add_observed_check(
                claim="verification: pytest",
                command="pytest",
                exit_code=0,
                task_id="test-task",
            )

            criteria = SuccessCriteria(
                task_id="test-task",
                description="run tests",
                criteria=[
                    Criterion(id="tests_pass", kind=CriterionKind.COMMAND_EXIT_CODE, command="pytest", expected=0),
                    Criterion(id="no_unknown", kind=CriterionKind.NO_UNKNOWN_CHANGES),
                ],
            )

            gate = VerificationGate(
                root=root,
                evidence=evidence,
                unknown_paths={"EXTERNAL.md"},
            )
            report = gate.evaluate(criteria)

            self.assertEqual(report.decision, Decision.BLOCKED)

    def test_gate_repair_when_test_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _setup_sandbox(root)

            evidence = EvidenceStore(root, task_id="test-task")
            evidence.add_inferred(
                claim="verification: pytest",
                details={"command": "pytest", "exit_code": 1, "kind": "verification_command"},
                task_id="test-task",
            )

            criteria = SuccessCriteria(
                task_id="test-task",
                description="run tests",
                criteria=[
                    Criterion(id="tests_pass", kind=CriterionKind.COMMAND_EXIT_CODE, command="pytest", expected=0),
                ],
                max_repairs=3,
            )

            gate = VerificationGate(root=root, evidence=evidence, repair_count=0)
            report = gate.evaluate(criteria, budget_spent=0)

            self.assertEqual(report.decision, Decision.REPAIR)

    def test_gate_rollback_when_budget_exhausted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _setup_sandbox(root)

            evidence = EvidenceStore(root, task_id="test-task")
            evidence.add_inferred(
                claim="verification: pytest",
                details={"command": "pytest", "exit_code": 1, "kind": "verification_command"},
                task_id="test-task",
            )

            criteria = SuccessCriteria(
                task_id="test-task",
                description="run tests",
                criteria=[
                    Criterion(id="tests_pass", kind=CriterionKind.COMMAND_EXIT_CODE, command="pytest", expected=0),
                ],
                max_repairs=3,
            )

            gate = VerificationGate(root=root, evidence=evidence, repair_count=3)
            report = gate.evaluate(criteria, budget_spent=0)

            self.assertEqual(report.decision, Decision.ROLLBACK)


# ────────────────────────────────────────────────────────────────────────
# Failure signature streak tests
# ────────────────────────────────────────────────────────────────────────


class FailureSignatureStreakTests(unittest.TestCase):
    """Failure signature streak detection."""

    def test_same_signature_produces_same_hash(self):
        report = Report(
            decision=Decision.REPAIR,
            results=[Result(criterion_id="c1", status="FAIL", observed=1)],
        )
        sig1 = FailureSignature.from_report(report, ["file.py"])
        sig2 = FailureSignature.from_report(report, ["file.py"])
        self.assertEqual(sig1.signature, sig2.signature)
        self.assertEqual(sig1.file_set, sig2.file_set)

    def test_different_observed_produces_different_hash(self):
        r1 = Report(
            decision=Decision.REPAIR,
            results=[Result(criterion_id="c1", status="FAIL", observed=1)],
        )
        r2 = Report(
            decision=Decision.REPAIR,
            results=[Result(criterion_id="c1", status="FAIL", observed=2)],
        )
        sig1 = FailureSignature.from_report(r1, ["file.py"])
        sig2 = FailureSignature.from_report(r2, ["file.py"])
        self.assertNotEqual(sig1.signature, sig2.signature)

    def test_streak_increments(self):
        report = Report(
            decision=Decision.REPAIR,
            results=[Result(criterion_id="c1", status="FAIL", observed=1)],
        )
        sig = FailureSignature.from_report(report, ["file.py"])
        sig.no_improvement_streak = 0

        sig2 = FailureSignature.from_report(report, ["file.py"])
        if sig.signature == sig2.signature and sig.file_set == sig2.file_set:
            sig2.no_improvement_streak = sig.no_improvement_streak + 1

        self.assertEqual(sig2.no_improvement_streak, 1)

    def test_streak_triggers_rollback_at_2(self):
        report = Report(
            decision=Decision.REPAIR,
            results=[Result(criterion_id="c1", status="FAIL", observed=1)],
        )

        streak = 0
        last_sig = None
        for _ in range(3):
            sig = FailureSignature.from_report(report, ["file.py"])
            if last_sig is not None and sig.signature == last_sig.signature and sig.file_set == last_sig.file_set:
                streak += 1
            sig.no_improvement_streak = streak
            last_sig = sig

        self.assertEqual(streak, 2)
        self.assertGreaterEqual(streak, 2)


# ────────────────────────────────────────────────────────────────────────
# FSM consistency tests
# ────────────────────────────────────────────────────────────────────────


class FSMConsistencyTests(unittest.TestCase):
    """FSM transition consistency."""

    def test_no_transitions_from_terminal(self):
        for state in (State.COMPLETED, State.REJECTED, State.ROLLED_BACK, State.FAILED):
            fsm = FSM(state)
            self.assertTrue(fsm.is_terminal())
            self.assertEqual(fsm.allowed_next(), set())

    def test_repairing_loop(self):
        fsm = FSM()
        fsm.transition(State.EXECUTING)   # 1
        fsm.transition(State.VERIFYING)   # 2
        fsm.transition(State.REPAIRING)   # 3
        fsm.transition(State.EXECUTING)   # 4
        fsm.transition(State.VERIFYING)   # 5
        self.assertEqual(fsm.state, State.VERIFYING)
        self.assertEqual(len(fsm.history), 5)

    def test_verifying_to_rolled_back(self):
        fsm = FSM()
        fsm.transition(State.EXECUTING)
        fsm.transition(State.VERIFYING)
        fsm.transition(State.ROLLED_BACK)
        self.assertTrue(fsm.is_terminal())

    def test_verifying_to_failed(self):
        fsm = FSM()
        fsm.transition(State.EXECUTING)
        fsm.transition(State.VERIFYING)
        fsm.transition(State.FAILED)
        self.assertTrue(fsm.is_terminal())


if __name__ == "__main__":
    unittest.main()
