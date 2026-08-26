"""Canonical event log and bounded display queue for UI4.

The canonical log is an append-only, deduplicated record of validated,
redacted events. The display queue is intentionally bounded and lossy: a slow
renderer may drop old display items, but it can never alter the canonical log.
Neither component makes security decisions or creates Evidence.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Iterable, Mapping, Optional


class CanonicalEventLog:
    """Append-only event history preserving arrival order and unique IDs."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._seen_ids: set[str] = set()

    def append(self, event: Mapping[str, Any]) -> bool:
        event_id = str(event.get("event_id", ""))
        if not event_id or event_id in self._seen_ids:
            return False
        self._seen_ids.add(event_id)
        self._events.append(dict(event))
        return True

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._events)

    def __len__(self) -> int:
        return len(self._events)


class BoundedDisplayQueue:
    """Bounded FIFO for renderers; dropping display items never drops history."""

    def __init__(self, maxsize: int = 256) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self._items: Deque[dict[str, Any]] = deque(maxlen=maxsize)
        self.dropped_count = 0

    def put(self, event: Mapping[str, Any]) -> None:
        if len(self._items) == self.maxsize:
            self.dropped_count += 1
        self._items.append(dict(event))

    def get_nowait(self) -> Optional[dict[str, Any]]:
        if not self._items:
            return None
        return self._items.popleft()

    def drain(self) -> list[dict[str, Any]]:
        items = list(self._items)
        self._items.clear()
        return items

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._items)

    def __len__(self) -> int:
        return len(self._items)
