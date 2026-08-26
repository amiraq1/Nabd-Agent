"""Secret redaction for artifacts written to disk (evidence, rollback logs).

Any value that looks like a credential (assignment-style secret, Bearer
token, or an OpenAI ``sk-`` key) is replaced with ``[REDACTED]`` before the
data leaves memory and is persisted.  This prevents accidental leakage of
secrets into ``.nabd/evidence.json`` or rollback logs.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# Assignment-style secrets:  KEY=VALUE  or  KEY: VALUE
_ASSIGN_RE = re.compile(
    r"(?i)((?:api[_-]?key|apikey|secret[_-]?key|access[_-]?key|"
    r"private[_-]?key|client[_-]?secret|secret|token|password|passwd|pwd|"
    r"authorization)\s*[=:]\s*)([^\s'\"`,;<>]+)"
)
# HTTP Authorization: Bearer <token>
_BEARER_RE = re.compile(r"(?i)(bearer\s+)([A-Za-z0-9\-._~+/]+=*)")
# OpenAI-style secret keys: sk-<base64-ish>
_SK_RE = re.compile(r"\b(sk-[A-Za-z0-9]{16,})\b")


def redact_text(text: str) -> str:
    """Return *text* with any detected secret replaced by ``[REDACTED]``."""
    if not isinstance(text, str):
        return text
    text = _ASSIGN_RE.sub(lambda m: m.group(1) + "[REDACTED]", text)
    text = _BEARER_RE.sub(lambda m: m.group(1) + "[REDACTED]", text)
    text = _SK_RE.sub("[REDACTED]", text)
    return text


def redact_obj(value: Any) -> Any:
    """Recursively redact every string value inside a JSON-like structure."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return type(value)(redact_obj(v) for v in value)
    return value


def redact_pairs(pairs: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    """Convenience for dict(pairs) post-redaction (e.g. asdict hooks)."""
    return [(k, redact_obj(v)) for k, v in pairs]
