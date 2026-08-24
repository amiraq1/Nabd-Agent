"""Standalone file verifier for Nabd, implemented with Python stdlib."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .jail import JailError, WorkspaceJail


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_path(jail: WorkspaceJail, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    return candidate if candidate.is_absolute() else jail.workspace_root / candidate


def safe_file(jail: WorkspaceJail, raw_path: str) -> Path:
    try:
        target = jail.check_path(_workspace_path(jail, raw_path), allow_missing=False)
    except JailError as exc:
        raise ValueError(str(exc)) from exc
    if not target.is_file():
        raise ValueError(f"Not a file: {raw_path}")
    return target


def backup_file(jail: WorkspaceJail, target: Path, backup_dir: str | None = None) -> Path:
    destination_dir = jail.check_path(
        _workspace_path(jail, backup_dir or ".nabd/backups")
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = destination_dir / f"{target.name}.backup.{timestamp}"
    shutil.copy2(target, destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nabd-verify",
        description="Nabd file verification and workspace safety utility",
    )
    parser.add_argument("--root", default=".", help="workspace root, default: current directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hash_parser = subparsers.add_parser("hash", help="calculate SHA-256")
    hash_parser.add_argument("path")

    verify_parser = subparsers.add_parser("verify", help="verify file SHA-256")
    verify_parser.add_argument("path")
    verify_parser.add_argument("expected_hash")

    jail_parser = subparsers.add_parser("jail", help="check a path inside workspace")
    jail_parser.add_argument("path")

    backup_parser = subparsers.add_parser("backup", help="create a timestamped backup")
    backup_parser.add_argument("path")
    backup_parser.add_argument("--dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = Path(args.root).expanduser().resolve()
        jail = WorkspaceJail(root)
        if args.command == "jail":
            jail.check_path(args.path)
            print("SAFE")
            return 0
        target = safe_file(jail, args.path)
        if args.command == "hash":
            print(compute_sha256(target))
            return 0
        if args.command == "verify":
            expected = args.expected_hash.strip().lower()
            if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
                raise ValueError("expected_hash must be a 64-character SHA-256 hex string")
            actual = compute_sha256(target)
            if actual != expected:
                print("MISMATCH")
                print(f"expected: {expected}", file=sys.stderr)
                print(f"actual:   {actual}", file=sys.stderr)
                return 1
            print("OBSERVED")
            return 0
        if args.command == "backup":
            print(backup_file(jail, target, args.dir))
            return 0
        raise ValueError(f"Unknown command: {args.command}")
    except (JailError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
