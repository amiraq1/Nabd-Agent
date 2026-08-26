"""Thin runtime shim binding the UI/Renderer to the real NabdAgent.

This module replaces the UI-3 standalone ``Runtime.run_demo`` entry point with
``run_task``. It does NOT alter agent behaviour: it only constructs a real
:class:`NabdAgent` (optionally with an event sink and an approval callback) and
delegates to :meth:`NabdAgent.run`. The standalone UI-3 demo stays separate and
must not be invoked from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .agent import NabdAgent
from .approval import ApprovalMode, coerce_approval_mode
from .event_contract import EventBoundary
from .events import AgentEventSink, AgentRunResult, TeeEventSink


def run_task(
    task: str = "ارفع ملخصًا موجزًا لبنية المشروع الحالي",
    root: str | Path = ".",
    provider: str = "auto",
    max_rounds: int = 3,
    workspace_free: bool = False,
    llm_client: Optional[Any] = None,
    event_sink: Optional[AgentEventSink] = None,
    approval_callback: Optional[Callable[[Mapping[str, Any]], bool]] = None,
    *,
    approval_mode: ApprovalMode | str,
) -> AgentRunResult:
    """Run a single agent task and stream events to the Renderer.

    Thin wrapper over :meth:`NabdAgent.run`; core logic is untouched. Pass a
        fake client (e.g. ``helpers.fake_llm.FakeLLMClient``) to keep it offline and deterministic for tests. ``approval_mode`` is mandatory: callers must
    explicitly choose ``CONFIRM`` or ``AUTO``; there is no implicit AUTO path.


    Streaming: events are published live to ``event_sink`` (the Rich renderer)
    and also captured in the returned ``AgentRunResult.events`` list.
    """
    mode = coerce_approval_mode(approval_mode)
    capturer = TeeEventSink(event_sink)
    # The boundary validates + redacts every event before it reaches the
    # Renderer (or the capturer that records it for tests).
    boundary = EventBoundary(capturer)
    agent = NabdAgent(
        root,
        provider=provider,
        auto_approve=mode is ApprovalMode.AUTO,
        workspace_free=workspace_free,
        approval_mode=mode,
        llm_client=llm_client,
        event_sink=boundary,
        approval_callback=approval_callback,
    )
    result = agent.run(task, max_rounds=max(1, min(max_rounds, 10)))
    return AgentRunResult(
        state=result.state,
        summary=result.summary,
        changes=result.changes,
        evidence=result.evidence,
        error=result.error,
        ok=result.ok,
        events=capturer.events,
    )
