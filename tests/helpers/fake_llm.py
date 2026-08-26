"""TEST-ONLY offline fake LLM client for Nabd.

This module lives under ``tests/`` and is NEVER imported by production code.
It lets the agent execute a deterministic, valid controlled-mutation plan
without any real LLM provider, API key, or network access.

The fake returns a fixed plan that uses only agent tools
(``read_file`` / ``write_file``) and a read-only verification command. It
never emits a human-language string as a ``run_command``, and it performs
no I/O other than reading the fixture README to append the marker line.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


MARKER = "Controlled mutation marker: Nabd M1 smoke."


class FakeLLMClient:
    """Deterministic stand-in for :class:`nabd.llm.LLMClient`.

    Used exclusively by Python tests. It records every ``complete_json``
    call so tests can prove the real provider was never contacted.
    """

    # Quack like LLMClient for diagnostics; never used for network calls.
    provider = "fake"
    model = "fake-test"

    def __init__(
        self,
        root: Any = ".",
        marker: str = MARKER,
        plan_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.root = Path(root)
        self.marker = marker
        self._plan_override = plan_override
        self.calls: list = []

    def _readme_mutation_plan(self) -> Dict[str, Any]:
        readme = self.root / "README.md"
        original = readme.read_text(encoding="utf-8") if readme.exists() else ""
        if original and not original.endswith("\n"):
            original += "\n"
        new_content = original + self.marker + "\n"
        return {
            "summary": "Controlled mutation test",
            "steps": [
                "قراءة README.md",
                "إضافة marker واحد في نهاية README.md",
                "تشغيل اختبار fixture",
            ],
            "actions": [
                {"name": "read_file", "arguments": {"path": "README.md"}},
                {
                    "name": "write_file",
                    "arguments": {"path": "README.md", "content": new_content},
                },
            ],
            "verification": ["python -m unittest discover -s tests -v"],
        }

    def complete_json(self, system: str, user: str) -> Dict[str, Any]:
        self.calls.append((system, user))
        if self._plan_override is not None:
            return self._plan_override
        return self._readme_mutation_plan()

    def complete(self, system: str, user: str) -> str:
        return json.dumps(self.complete_json(system, user))
