import hashlib
import tempfile
import unittest
from pathlib import Path

from nabd.jail import WorkspaceJail
from nabd.verifier import backup_file, compute_sha256, safe_file


class VerifierTests(unittest.TestCase):
    def test_hash_and_safe_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "sample.txt"
            target.write_text("abc\n", encoding="utf-8")
            self.assertEqual(
                compute_sha256(target),
                hashlib.sha256(b"abc\n").hexdigest(),
            )
            self.assertEqual(safe_file(WorkspaceJail(root), "sample.txt"), target)

    def test_backup_is_created_inside_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "sample.txt"
            target.write_text("abc\n", encoding="utf-8")
            backup = backup_file(WorkspaceJail(root), target)
            self.assertTrue(backup.is_file())
            self.assertEqual(backup.read_text(encoding="utf-8"), "abc\n")
            self.assertIn(".nabd/backups", str(backup.relative_to(root)))


if __name__ == "__main__":
    unittest.main()
