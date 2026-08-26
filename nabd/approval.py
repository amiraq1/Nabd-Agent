"""Core-owned approval policy and synchronous decision providers."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Mapping, Protocol


class ApprovalMode(str, Enum):
    """Explicit approval policy selected by the caller."""

    CONFIRM = "CONFIRM"
    AUTO = "AUTO"


def coerce_approval_mode(value: ApprovalMode | str) -> ApprovalMode:
    if isinstance(value, ApprovalMode):
        return value
    try:
        return ApprovalMode(str(value).strip().upper())
    except ValueError as exc:
        raise ValueError("approval_mode must be CONFIRM or AUTO") from exc


class ApprovalProvider(Protocol):
    """Synchronous, core-invoked decision provider."""

    def decide(self, request: Mapping[str, Any]) -> bool:
        ...


class CallbackApprovalProvider:
    """Adapter for a caller-owned synchronous callback."""

    def __init__(self, callback: Callable[[Mapping[str, Any]], bool]) -> None:
        self._callback = callback

    def decide(self, request: Mapping[str, Any]) -> bool:
        try:
            return bool(self._callback(request))
        except Exception:
            # Renderer/callback failure is fail-closed.
            return False


class InteractiveApprovalProvider:
    """Terminal provider with an explicit allowlist and default deny."""

    _ALLOW = frozenset({"y", "yes", "نعم"})

    def decide(self, request: Mapping[str, Any]) -> bool:
        print(f"\nطلب الوكيل تنفيذ: {request.get('display', request.get('tool', 'unknown'))}")
        try:
            answer = input("السماح؟ [y/N]: ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in self._ALLOW
