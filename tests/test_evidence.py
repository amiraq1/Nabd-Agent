import tempfile
import unittest
from pathlib import Path

from nabd.evidence import EvidenceStore, EvidenceType
from nabd.raw_facts import RawFacts


class EvidenceTests(unittest.TestCase):
    def test_observed_file_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "answer.txt"
            target.write_text("verified\n", encoding="utf-8")
            store = EvidenceStore(root)
            store.add_observed("answer exists", "answer.txt")
            self.assertTrue(store.all_observed())
            target.write_text("changed\n", encoding="utf-8")
            self.assertFalse(store.all_observed())

    def test_inferred_evidence_never_proves_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "answer.txt"
            target.write_text("verified\n", encoding="utf-8")
            store = EvidenceStore(root)
            store.add_observed("answer exists", "answer.txt")
            store.add_inferred("the feature probably works")
            self.assertFalse(store.all_observed())

    def test_successful_command_can_be_observed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(Path(directory))
            store.add_observed_check("check passed", "python -m compileall", 0, "ok")
            self.assertTrue(store.all_observed())

    def test_empty_store_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(EvidenceStore(Path(directory)).all_observed())

    def test_inferred_details_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(Path(directory))
            evidence = store.add_inferred("thought", {"reason": "not verified"})
            self.assertEqual(evidence.details["reason"], "not verified")

    def test_observed_requires_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(Path(directory))
            with self.assertRaises(FileNotFoundError):
                store.add_observed("missing", "nonexistent/file.txt")

    def test_compute_hash_has_sha256_length(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "sample.txt"
            target.write_text("test", encoding="utf-8")
            digest = EvidenceStore.compute_file_hash(target)
            self.assertEqual(len(digest), 64)

    def test_evidence_filters_and_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "sample.txt"
            target.write_text("hello", encoding="utf-8")
            store = EvidenceStore(root)
            observed = store.add_observed("file exists", "sample.txt")
            inferred = store.add_inferred("probably complete")
            self.assertEqual(store.get_observed(), [observed])
            self.assertEqual(store.get_inferred(), [inferred])
            payload = observed.to_dict()
            self.assertEqual(payload["type"], "OBSERVED")
            self.assertNotIn("evidence_type", payload)

    def test_string_root_and_hash_path_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "sample.txt"
            target.write_text("test", encoding="utf-8")
            store = EvidenceStore(directory)
            observed = store.add_observed("file exists", "sample.txt")
            self.assertEqual(observed.sha256, EvidenceStore.compute_file_hash(str(target)))
            self.assertTrue(store.all_observed())

    def test_raw_facts_require_current_task_and_relevance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "answer.txt"
            target.write_text("verified", encoding="utf-8")
            raw = RawFacts(
                operation="read", path="answer.txt", exists=True,
                size=8, sha256=EvidenceStore.compute_file_hash(target),
                mtime=target.stat().st_mtime,
            )
            store = EvidenceStore(root, task_id="current")
            old = store.verify(raw, task_id="old")
            self.assertEqual(old.evidence_type, EvidenceType.INFERRED)
            relevant = store.verify(raw, task_id="current", relevant=False)
            self.assertEqual(relevant.evidence_type, EvidenceType.INFERRED)
            self.assertFalse(store.is_usable_for_completion("current"))

    def test_stale_raw_facts_are_not_observed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "answer.txt"
            target.write_text("verified", encoding="utf-8")
            raw = RawFacts(
                operation="read", path="answer.txt", exists=True,
                size=8, sha256=EvidenceStore.compute_file_hash(target), mtime=0,
            )
            store = EvidenceStore(root, task_id="current")
            evidence = store.verify(raw, task_id="current", max_age_seconds=1)
            self.assertEqual(evidence.evidence_type, EvidenceType.INFERRED)
            self.assertFalse(evidence.fresh)

    def test_multiple_observed_files_must_all_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("one", encoding="utf-8")
            second.write_text("two", encoding="utf-8")
            store = EvidenceStore(root)
            store.add_observed("first exists", "first.txt")
            store.add_observed("second exists", "second.txt")
            self.assertTrue(store.all_observed())
            second.write_text("tampered", encoding="utf-8")
            self.assertFalse(store.all_observed())


if __name__ == "__main__":
    unittest.main()
