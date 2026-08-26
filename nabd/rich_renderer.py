"""Rich presentation layer for Nabd AgentEvents.

This module is deliberately presentation-only. RichEventSink receives core
mappings, routes them through EventBoundary and EventAdapter, then renders the
result. It never executes tools, creates Evidence, or decides PASS/BLOCKED.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from .event_contract import EventBoundary
from .event_stream import BoundedDisplayQueue, CanonicalEventLog
from .ui_adapter import EventAdapter, UiEvent


_STATE_STYLE = {
    "PLANNING": ("magenta", "Preparing a safe plan"),
    "EXECUTING": ("cyan", "Executing an approved tool"),
    "VERIFYING": ("yellow", "Verifying success criteria"),
    "REPAIRING": ("dark_orange", "Attempting a bounded repair"),
    "COMPLETED": ("green", "Task completed with evidence"),
    "REJECTED": ("red", "Rejected by policy"),
    "FAILED": ("red", "Task failed"),
    "ROLLED_BACK": ("magenta", "Restored to snapshot"),
}


class RichRenderer:
    """Pure presentation state for already-adapted, display-safe events."""

    def __init__(self, console: Optional[Console] = None, max_events: int = 12) -> None:
        self.console = console or Console()
        self.max_events = max(1, max_events)
        self.events: list[UiEvent] = []
        self.state = "PLANNING"
        self.status = _STATE_STYLE[self.state][1]

    def ingest(self, event: UiEvent) -> None:
        self.events.append(event)
        self.events = self.events[-self.max_events :]
        self.state = event.fsm_state or self.state
        fallback = _STATE_STYLE.get(self.state, ("white", self.state))[1]
        self.status = event.summary or fallback

    def ingest_many(self, events: Iterable[UiEvent]) -> None:
        for event in events:
            self.ingest(event)

    def _clip(self, value: Any, limit: int = 88) -> str:
        text = " ".join(str(value).replace("\n", " ").split())
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"

    def render_header(self) -> Panel:
        color, fallback = _STATE_STYLE.get(self.state, ("white", self.state))
        evidence_count = sum(1 for event in self.events if event.evidence_id)
        body = Text.assemble(
            ("Nabd", "bold white"),
            (f"  {self.state}\n", f"bold {color}"),
            (self._clip(self.status, 84) + "\n", "bright_white"),
            (f"events={len(self.events)}  evidence={evidence_count}", "dim"),
        )
        return Panel(body, border_style=color)

    def render_event(self, event: UiEvent) -> RenderableType:
        color, _ = _STATE_STYLE.get(event.fsm_state or "", ("white", ""))
        label = event.event_type.replace("_", " ")
        details: list[str] = [f"attempt={event.attempt_order} seq={event.seq}"]
        if event.tool_name:
            details.append(f"tool={event.tool_name}")
        if event.file_path:
            details.append(f"path={event.file_path}")
        if event.decision:
            details.append(f"decision={event.decision}")
        if event.approval_status:
            details.append(f"approval={event.approval_status}")
        if event.evidence_id:
            details.append(f"evidence={event.evidence_id}")
        summary = self._clip(event.summary)
        return Text.assemble(
            (f"{label:<24}", f"bold {color}"),
            (summary + "\n", "white"),
            ("  " + "  ".join(details), "dim"),
        )

    def render_input_station(self, current_input: str = "") -> Text:
        return Text.assemble(
            ("━" * 72 + "\n", "bright_white"),
            ("› ", "bold magenta"),
            (current_input or "Ask your question...", "bright_magenta" if current_input else "dim"),
            ("\n" + "━" * 72, "bright_white"),
        )

    def render(self, current_input: str = "") -> Group:
        rows: list[RenderableType] = [self.render_header()]
        if self.events:
            rows.extend(self.render_event(event) for event in self.events)
        else:
            rows.append(Text("Nabd is ready", style="dim"))
        rows.append(self.render_input_station(current_input))
        return Group(*rows)


class _RendererDelegate:
    """Small publish-compatible delegate used only inside RichEventSink."""

    def __init__(self, owner: "RichEventSink") -> None:
        self.owner = owner

    def publish(self, event: Mapping[str, Any]) -> None:
        self.owner._consume(event)


class RichEventSink:
    """Core-to-Rich sink: Boundary -> Adapter -> Renderer."""

    def __init__(
        self,
        console: Optional[Console] = None,
        max_events: int = 12,
        queue_size: int = 256,
        live: bool = False,
    ) -> None:
        self.console = console or Console()
        self.renderer = RichRenderer(self.console, max_events=max_events)
        self.adapter = EventAdapter()
        self.canonical_log = CanonicalEventLog()
        self.display_queue = BoundedDisplayQueue(maxsize=queue_size)
        self.live = live
        self.boundary = EventBoundary(
            _RendererDelegate(self),
            canonical_log=self.canonical_log,
            display_queue=self.display_queue,
        )

    def publish(self, event: Mapping[str, Any]) -> None:
        self.boundary.publish(event)

    def _consume(self, event: Mapping[str, Any]) -> None:
        projected = self.adapter.adapt(event)
        self.renderer.ingest(projected)
        if self.live:
            self.console.print(self.renderer.render_event(projected))

    def render(self, current_input: str = "") -> Group:
        return self.renderer.render(current_input)
