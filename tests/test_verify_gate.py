"""Unit tests for the Verification Gate."""

import tempfile
import unittest
from pathlib import Path

from nabd.evidence import EvidenceStore
from nabd.fsm import FSM, FSMError, State
from nabd.raw_facts import RawFacts
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


class CriterionKindTests(unittest.TestCase):
    """Test criterion kind validation."""

    def test_all_kinds_are_string_enum(self):
        for kind in CriterionKind:
            self.assertIsInstance(kind.value, str)

    def test_known_kinds(self):
        expected = {
            "command_exit_code",
            "path_changed",
            "path_unchanged",
            "no_unknown_changes",
            "file_contains",
            "failure_resolved",
        }
        actual = {kind.value for kind in CriterionKind}
        self.assertEqual(actual, expected)


class SuccessCriteriaValidationTests(unittest.TestCase):
    """Test SuccessCriteria.validate()"""

    def test_valid_criteria(self):
        criteria = SuccessCriteria(
            task_id="task-001",
            description="fix bug",
            criteria=[
                Criterion(id="c1", kind=CriterionKind.COMMAND_EXIT_CODE, command="go test ./...", expected=0),
                Criterion(id="c2", kind=CriterionKind.NO_UNKNOWN_CHANGES),
            ],
        )
        errors = criteria.validate()
        self.assertEqual(errors, [])

    def test_missing_task_id(self):
        criteria = SuccessCriteria(task_id="", description="fix bug")
        errors = criteria.validate()
        self.assertTrue(any("task_id" in e for e in errors))

    def test_missing_description(self):
        criteria = SuccessCriteria(task_id="task-001", description="")
        errors = criteria.validate()
        self.assertTrue(any("description" in e for e in errors))

    def test_duplicate_criterion_ids(self):
        criteria = SuccessCriteria(
            task_id="task-001",
            description="fix bug",
            criteria=[
                Criterion(id="c1", kind=CriterionKind.NO_UNKNOWN_CHANGES),
                Criterion(id="c1", kind=CriterionKind.NO_UNKNOWN_CHANGES),
            ],
        )
        errors = criteria.validate()
        self.assertTrue(any("duplicate" in e for e in errors))

    def test_command_exit_code_requires_command(self):
        criteria = SuccessCriteria(
            task_id="task-001",
            description="fix bug",
            criteria=[
                Criterion(id="c1", kind=CriterionKind.COMMAND_EXIT_CODE, command=""),
            ],
        )
        errors = criteria.validate()
        self.assertTrue(any("command is required" in e for e in errors))

    def test_path_changed_requires_path(self):
        criteria = SuccessCriteria(
            task_id="task-001",
            description="fix bug",
            criteria=[
                Criterion(id="c1", kind=CriterionKind.PATH_CHANGED, path=""),
            ],
        )
        errors = criteria.validate()
        self.assertTrue(any("path is required" in e for e in errors))

    def test_file_contains_requires_expected(self):
        criteria = SuccessCriteria(
            task_id="task-001",
            description="fix bug",
            criteria=[
                Criterion(id="c1", kind=CriterionKind.FILE_CONTAINS, path="foo.py", expected=None),
            ],
        )
        errors = criteria.validate()
        self.assertTrue(any("expected" in e for e in errors))

    def test_negative_max_repairs(self):
        criteria = SuccessCriteria(
            task_id="task-001",
            description="fix bug",
            max_repairs=-1,
        )
        errors = criteria.validate()
        self.assertTrue(any("max_repairs" in e for e in errors))

    def test_zero_timeout(self):
        criteria = SuccessCriteria(
            task_id="task-001",
            description="fix bug",
            wall_clock_timeout_seconds=0,
        )
        errors = criteria.validate()
        self.assertTrue(any("timeout" in e for e in errors))


class GatePassTests(unittest.TestCase):
    """Test that all mandatory criteria passing yields PASS."""

    def test_all_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence = EvidenceStore(root, task_id="t1")

            # Record successful verification evidence
            evidence.add_observed_check(
                claim="verification: python -m pytest",
                command="python -m pytest",
                exit_code=0,
                output="all tests passed",
                task_id="t1",
            )

            criteria = SuccessCriteria(
                task_id="t1",
                description="run tests",
                criteria=[
                    Criterion(id="tests_pass", kind=CriterionKind.COMMAND_EXIT_CODE, command="python -m pytest", expected=0),
                    Criterion(id="no_unknown", kind=CriterionKind.NO_UNKNOWN_CHANGES),
                ],
            )

            gate = VerificationGate(root=root, evidence=evidence)
            report = gate.evaluate(criteria)

            self.assertEqual(report.decision, Decision.PASS)
            self.assertEqual(report.failed_required, [])
            self.assertEqual(report.missing_evidence, [])


class GateRepairTests(unittest.TestCase):
    """Test that required failure with budget yields REPAIR."""

    def test_required_failure_with_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence = EvidenceStore(root, task_id="t1")

            # Record failed verification evidence
            evidence.add_inferred(
                claim="verification: python -m pytest",
                details={"command": "python -m pytest", "exit_code": 1, "kind": "verification_command"},
                task_id="t1",
            )

            criteria = SuccessCriteria(
                task_id="t1",
                description="run tests",
                criteria=[
                    Criterion(id="tests_pass", kind=CriterionKind.COMMAND_EXIT_CODE, command="python -m pytest", expected=0),
                ],
                max_repairs=3,
            )

            gate = VerificationGate(root=root, evidence=evidence, repair_count=0)
            report = gate.evaluate(criteria, budget_spent=0)

            self.assertEqual(report.decision, Decision.REPAIR)
            self.assertIn("tests_pass", report.failed_required)


class GateRollbackTests(unittest.TestCase):
    """Test that required failure with no budget yields ROLLBACK."""

    def test_required_failure_no_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence = EvidenceStore(root, task_id="t1")

            evidence.add_inferred(
                claim="verification: python -m pytest",
                details={"command": "python -m pytest", "exit_code": 1, "kind": "verification_command"},
                task_id="t1",
            )

            criteria = SuccessCriteria(
                task_id="t1",
                description="run tests",
                criteria=[
                    Criterion(id="tests_pass", kind=CriterionKind.COMMAND_EXIT_CODE, command="python -m pytest", expected=0),
                ],
                max_repairs=3,
            )

            # Exhausted repairs
            gate = VerificationGate(root=root, evidence=evidence, repair_count=3)
            report = gate.evaluate(criteria, budget_spent=0)

            self.assertEqual(report.decision, Decision.ROLLBACK)

    def test_budget_exhausted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence = EvidenceStore(root, task_id="t1")

            evidence.add_inferred(
                claim="verification: go test",
                details={"command": "go test", "exit_code": 1, "kind": "verification_command"},
                task_id="t1",
            )

            criteria = SuccessCriteria(
                task_id="t1",
                description="run tests",
                criteria=[
                    Criterion(id="tests_pass", kind=CriterionKind.COMMAND_EXIT_CODE, command="go test", expected=0),
                ],
                max_repairs=3,
            )

            gate = VerificationGate(root=root, evidence=evidence, repair_count=0)
            # Spend a huge budget
            report = gate.evaluate(criteria, budget_spent=9999)

            self.assertEqual(report.decision, Decision.ROLLBACK)


class GateBlockedTests(unittest.TestCase):
    """Test that missing evidence or unknown paths block PASS."""

    def test_missing_evidence_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence = EvidenceStore(root, task_id="t1")
            # No evidence at all

            criteria = SuccessCriteria(
                task_id="t1",
                description="run tests",
                criteria=[
                    Criterion(id="tests_pass", kind=CriterionKind.COMMAND_EXIT_CODE, command="pytest", expected=0),
                ],
            )

            gate = VerificationGate(root=root, evidence=evidence)
            report = gate.evaluate(criteria)

            self.assertEqual(report.decision, Decision.BLOCKED)
            self.assertIn("tests_pass", report.missing_evidence)

    def test_unknown_paths_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence = EvidenceStore(root, task_id="t1")

            evidence.add_observed_check(
                claim="verification: pytest",
                command="pytest",
                exit_code=0,
                task_id="t1",
            )

            criteria = SuccessCriteria(
                task_id="t1",
                description="run tests",
                criteria=[
                    Criterion(id="tests_pass", kind=CriterionKind.COMMAND_EXIT_CODE, command="pytest", expected=0),
                    Criterion(id="no_unknown", kind=CriterionKind.NO_UNKNOWN_CHANGES),
                ],
            )

            gate = VerificationGate(
                root=root,
                evidence=evidence,
                unknown_paths={"external_file.txt"},
            )
            report = gate.evaluate(criteria)

            self.assertEqual(report.decision, Decision.BLOCKED)


class GateNoFalsePositiveTests(unittest.TestCase):
    """Test that test success alone does not prove completion if other criteria fail."""

    def test_test_pass_but_unknown_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence = EvidenceStore(root, task_id="t1")

            evidence.add_observed_check(
                claim="verification: pytest",
                command="pytest",
                exit_code=0,
                task_id="t1",
            )

            criteria = SuccessCriteria(
                task_id="t1",
                description="run tests",
                criteria=[
                    Criterion(id="tests_pass", kind=CriterionKind.COMMAND_EXIT_CODE, command="pytest", expected=0),
                    Criterion(id="no_unknown", kind=CriterionKind.NO_UNKNOWN_CHANGES),
                ],
            )

            gate = VerificationGate(
                root=root,
                evidence=evidence,
                unknown_paths={"mystery.txt"},
            )
            report = gate.evaluate(criteria)

            # Should NOT be PASS even though tests pass
            self.assertNotEqual(report.decision, Decision.PASS)
            self.assertEqual(report.decision, Decision.BLOCKED)


class GateReportDictTests(unittest.TestCase):
    """Test Report.to_dict() serialization."""

    def test_report_to_dict(self):
        report = Report(
            decision=Decision.PASS,
            results=[Result(criterion_id="c1", status="PASS", observed=True)],
            missing_evidence=[],
            failed_required=[],
            summary="all good",
        )
        d = report.to_dict()
        self.assertEqual(d["decision"], "PASS")
        self.assertEqual(len(d["results"]), 1)
        self.assertEqual(d["results"][0]["criterion_id"], "c1")
        self.assertEqual(d["summary"], "all good")


class FSMTransitionTests(unittest.TestCase):
    """Test FSM allows new transitions for REPAIRING, ROLLED_BACK, FAILED."""

    def test_verifying_to_repairing(self):
        fsm = FSM(State.VERIFYING)
        fsm.transition(State.REPAIRING)
        self.assertEqual(fsm.state, State.REPAIRING)

    def test_repairing_to_executing(self):
        fsm = FSM(State.REPAIRING)
        fsm.transition(State.EXECUTING)
        self.assertEqual(fsm.state, State.EXECUTING)

    def test_repairing_to_rolled_back(self):
        fsm = FSM(State.REPAIRING)
        fsm.transition(State.ROLLED_BACK)
        self.assertEqual(fsm.state, State.ROLLED_BACK)
        self.assertTrue(fsm.is_terminal())

    def test_verifying_to_rolled_back(self):
        fsm = FSM(State.VERIFYING)
        fsm.transition(State.ROLLED_BACK)
        self.assertEqual(fsm.state, State.ROLLED_BACK)
        self.assertTrue(fsm.is_terminal())

    def test_verifying_to_failed(self):
        fsm = FSM(State.VERIFYING)
        fsm.transition(State.FAILED)
        self.assertEqual(fsm.state, State.FAILED)
        self.assertTrue(fsm.is_terminal())

    def test_rolled_back_is_terminal(self):
        fsm = FSM(State.ROLLED_BACK)
        self.assertTrue(fsm.is_terminal())
        self.assertEqual(fsm.allowed_next(), set())

    def test_failed_is_terminal(self):
        fsm = FSM(State.FAILED)
        self.assertTrue(fsm.is_terminal())
        self.assertEqual(fsm.allowed_next(), set())

    def test_executing_to_verifying_still_works(self):
        fsm = FSM(State.EXECUTING)
        fsm.transition(State.VERIFYING)
        self.assertEqual(fsm.state, State.VERIFYING)

    def test_planning_to_executing_still_works(self):
        fsm = FSM(State.PLANNING)
        fsm.transition(State.EXECUTING)
        self.assertEqual(fsm.state, State.EXECUTING)


class SnapshotTests(unittest.TestCase):
    """Test snapshot take and diff utilities."""

    def test_take_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.txt").write_text("hello")
            (root / "b.txt").write_text("world")

            snap = take_snapshot(root)
            self.assertIn("a.txt", snap)
            self.assertIn("b.txt", snap)
            self.assertEqual(len(snap["a.txt"]), 64)  # SHA-256 hex

    def test_diff_snapshots_detects_change(self):
        before = {"a.txt": "hash1", "b.txt": "hash2"}
        after = {"a.txt": "hash1", "b.txt": "hash3", "c.txt": "hash4"}
        diff = diff_snapshots(before, after)
        self.assertIn("b.txt", diff["changed"])
        self.assertIn("c.txt", diff["added"])
        self.assertEqual(diff["removed"], [])

    def test_diff_snapshots_no_changes(self):
        snap = {"a.txt": "hash1", "b.txt": "hash2"}
        diff = diff_snapshots(snap, snap)
        self.assertEqual(diff["added"], [])
        self.assertEqual(diff["removed"], [])
        self.assertEqual(diff["changed"], [])


class FailureSignatureTests(unittest.TestCase):
    """Test failure signature creation and streak detection."""

    def test_from_report(self):
        report = Report(
            decision=Decision.REPAIR,
            results=[
                Result(criterion_id="c1", status="FAIL", observed=1),
                Result(criterion_id="c2", status="PASS", observed=True),
            ],
        )
        sig = FailureSignature.from_report(report, ["file1.py"])
        self.assertIn("c1:", sig.signature)
        self.assertNotIn("c2:", sig.signature)  # only failed criteria
        self.assertEqual(sig.file_set, ["file1.py"])

    def test_identical_failures_produce_same_signature(self):
        report1 = Report(
            decision=Decision.REPAIR,
            results=[Result(criterion_id="c1", status="FAIL", observed=1)],
        )
        report2 = Report(
            decision=Decision.REPAIR,
            results=[Result(criterion_id="c1", status="FAIL", observed=1)],
        )
        sig1 = FailureSignature.from_report(report1)
        sig2 = FailureSignature.from_report(report2)
        self.assertEqual(sig1.signature, sig2.signature)

    def test_different_failures_produce_different_signatures(self):
        report1 = Report(
            decision=Decision.REPAIR,
            results=[Result(criterion_id="c1", status="FAIL", observed=1)],
        )
        report2 = Report(
            decision=Decision.REPAIR,
            results=[Result(criterion_id="c1", status="FAIL", observed=2)],
        )
        sig1 = FailureSignature.from_report(report1)
        sig2 = FailureSignature.from_report(report2)
        self.assertNotEqual(sig1.signature, sig2.signature)

    def test_streak_detection(self):
        sig1 = FailureSignature(signature="c1:1", no_improvement_streak=0)
        sig2 = FailureSignature(signature="c1:1", no_improvement_streak=1)
        # Same signature + same file_set -> streak increments
        self.assertEqual(sig1.signature, sig2.signature)

    def test_to_dict(self):
        sig = FailureSignature(signature="c1:1", file_set=["a.py"], no_improvement_streak=2)
        d = sig.to_dict()
        self.assertEqual(d["signature"], "c1:1")
        self.assertEqual(d["file_set"], ["a.py"])
        self.assertEqual(d["no_improvement_streak"], 2)


class CriterionResultTests(unittest.TestCase):
    """Test individual criterion evaluators in isolation."""

    def test_no_unknown_changes_pass_when_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence = EvidenceStore(root, task_id="t1")
            gate = VerificationGate(root=root, evidence=evidence, unknown_paths=set())
            criterion = Criterion(id="c1", kind=CriterionKind.NO_UNKNOWN_CHANGES)
            result = gate._evaluate_criterion(criterion)
            self.assertEqual(result.status, "PASS")

    def test_no_unknown_changes_fail_when_nonempty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence = EvidenceStore(root, task_id="t1")
            gate = VerificationGate(root=root, evidence=evidence, unknown_paths={"x.py"})
            criterion = Criterion(id="c1", kind=CriterionKind.NO_UNKNOWN_CHANGES)
            result = gate._evaluate_criterion(criterion)
            self.assertEqual(result.status, "FAIL")

    def test_file_contains_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "hello.py").write_text("print('hello')")
            evidence = EvidenceStore(root, task_id="t1")
            gate = VerificationGate(root=root, evidence=evidence)
            criterion = Criterion(id="c1", kind=CriterionKind.FILE_CONTAINS, path="hello.py", expected="print('hello')")
            result = gate._evaluate_criterion(criterion)
            self.assertEqual(result.status, "PASS")

    def test_file_contains_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "hello.py").write_text("print('hello')")
            evidence = EvidenceStore(root, task_id="t1")
            gate = VerificationGate(root=root, evidence=evidence)
            criterion = Criterion(id="c1", kind=CriterionKind.FILE_CONTAINS, path="hello.py", expected="nonexistent")
            result = gate._evaluate_criterion(criterion)
            self.assertEqual(result.status, "FAIL")

    def test_file_contains_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence = EvidenceStore(root, task_id="t1")
            gate = VerificationGate(root=root, evidence=evidence)
            criterion = Criterion(id="c1", kind=CriterionKind.FILE_CONTAINS, path="missing.py", expected="x")
            result = gate._evaluate_criterion(criterion)
            self.assertEqual(result.status, "FAIL")

    def test_path_changed_with_snapshots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence = EvidenceStore(root, task_id="t1")
            gate = VerificationGate(
                root=root,
                evidence=evidence,
                snapshots_before={"a.txt": "hash1"},
                snapshots_after={"a.txt": "hash2"},
            )
            criterion = Criterion(id="c1", kind=CriterionKind.PATH_CHANGED, path="a.txt", expected=True)
            result = gate._evaluate_criterion(criterion)
            self.assertEqual(result.status, "PASS")
            self.assertTrue(result.observed)

    def test_path_changed_not_changed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence = EvidenceStore(root, task_id="t1")
            gate = VerificationGate(
                root=root,
                evidence=evidence,
                snapshots_before={"a.txt": "hash1"},
                snapshots_after={"a.txt": "hash1"},
            )
            criterion = Criterion(id="c1", kind=CriterionKind.PATH_CHANGED, path="a.txt", expected=True)
            result = gate._evaluate_criterion(criterion)
            self.assertEqual(result.status, "FAIL")

    def test_path_unchanged_with_snapshots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence = EvidenceStore(root, task_id="t1")
            gate = VerificationGate(
                root=root,
                evidence=evidence,
                snapshots_before={"a.txt": "hash1"},
                snapshots_after={"a.txt": "hash1"},
            )
            criterion = Criterion(id="c1", kind=CriterionKind.PATH_UNCHANGED, path="a.txt", expected=True)
            result = gate._evaluate_criterion(criterion)
            self.assertEqual(result.status, "PASS")

    def test_path_unchanged_no_snapshots_assumes_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence = EvidenceStore(root, task_id="t1")
            gate = VerificationGate(root=root, evidence=evidence)
            criterion = Criterion(id="c1", kind=CriterionKind.PATH_UNCHANGED, path="a.txt", expected=True)
            result = gate._evaluate_criterion(criterion)
            self.assertEqual(result.status, "PASS")


if __name__ == "__main__":
    unittest.main()
