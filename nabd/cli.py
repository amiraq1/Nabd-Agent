"""Command-line interface for Nabd."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agent import NabdAgent
from .llm import LLMError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nabd",
        description="Nabd: وكيل برمجة احترافي يعمل داخل مجلد المشروع.",
    )
    parser.add_argument("task", nargs="?", help="المهمة البرمجية التي تريد تنفيذها")
    parser.add_argument(
        "--root", "--workspace", dest="root", default=".",
        help="جذر المشروع، الافتراضي هو المجلد الحالي",
    )
    parser.add_argument("--provider", choices=["auto", "openai", "gemini", "nvidia"], default=os.getenv("NABD_PROVIDER", "auto"))
    parser.add_argument("--max-rounds", type=int, default=5, help="الحد الأقصى لدورات الإصلاح")
    approval = parser.add_mutually_exclusive_group()
    approval.add_argument(
        "--yes", "--auto", dest="auto_approve", action="store_true", default=True,
        help="التنفيذ التلقائي دون طلب موافقة (الوضع الافتراضي)",
    )
    approval.add_argument(
        "--confirm", dest="auto_approve", action="store_false",
        help="طلب موافقة قبل كل كتابة أو أمر تشغيل",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.task:
        parser.error("اكتب المهمة بين علامتي اقتباس، مثل: nabd 'أصلح اختبار تسجيل الدخول'")
    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        parser.error(f"مجلد المشروع غير موجود: {root}")

    print(f"Nabd يعمل داخل: {root}")
    print(f"المزود: {args.provider} | التنفيذ التلقائي: {'نعم' if args.auto_approve else 'لا'}")
    try:
        agent = NabdAgent(root, provider=args.provider, auto_approve=args.auto_approve)
    except LLMError as exc:
        print(f"خطأ في إعداد المزود: {exc}", file=sys.stderr)
        return 2
    result = agent.run(args.task, max_rounds=max(1, min(args.max_rounds, 10)))

    print("\n===== النتيجة =====")
    print(f"الحالة: {result.state}")
    print(f"الملخص: {result.summary}")
    if result.changes:
        print("التغييرات:")
        for change in result.changes:
            print(f"- {change}")
    if result.error:
        print(f"الخطأ: {result.error}", file=sys.stderr)
    return 0 if result.ok else 1
