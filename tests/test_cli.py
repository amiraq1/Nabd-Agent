import unittest

from nabd.cli import build_parser


class CLITests(unittest.TestCase):
    def test_auto_approval_is_default(self):
        args = build_parser().parse_args(["مهمة تجريبية"])
        self.assertTrue(args.auto_approve)

    def test_confirm_disables_auto_approval(self):
        args = build_parser().parse_args(["مهمة تجريبية", "--confirm"])
        self.assertFalse(args.auto_approve)

    def test_legacy_yes_flag_stays_supported(self):
        args = build_parser().parse_args(["مهمة تجريبية", "--yes"])
        self.assertTrue(args.auto_approve)

    def test_workspace_alias_is_supported(self):
        args = build_parser().parse_args(["مهمة تجريبية", "--workspace", "/tmp/project"])
        self.assertEqual(args.root, "/tmp/project")


if __name__ == "__main__":
    unittest.main()
