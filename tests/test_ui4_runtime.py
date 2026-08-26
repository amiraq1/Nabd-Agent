"""UI-4 end-to-end runtime / event-binding tests.

These verify that ``run_task`` streams real intermediate events to the Renderer
and that the interactive approval path is owned by the Core (never auto-approved
by default). All runs use the offline ``FakeLLMClient``: no network, no real
provider, no secrets.
"""

from __future__ import annotations

from pathlib import Path
from nabd.agent import NabdAgent
from nabd.approval import ApprovalMode
from nabd.events import EventType
from nabd.runtime import AgentRunResult, run_task

from helpers.fake_llm import FakeLLMClient

FAIL_CMD = "false"
PASS_CMD = "true"


def _write_plan(content: str) -> dict:
    return {
        "summary": "ui4 write plan",
        "steps": ["read", "write", "verify"],
        "actions": [
            {"name": "read_file", "arguments": {"path": "README.md"}},
            {"name": "write_file", "arguments": {"path": "README.md", "content": content}},
        ],
        "verification": [PASS_CMD],
    }


def test_run_task_default_no_auto_approve(tmp_path: Path):
    (tmp_path / "README.md").write_text("orig\n", encoding="utf-8")
    approvals = []

    def cb(request):
        approvals.append(request)
        return False  # deny every mutating action

    result = run_task(
        "عدّل README",
        root=tmp_path,
        approval_mode=ApprovalMode.CONFIRM,
        llm_client=FakeLLMClient(tmp_path, plan_override=_write_plan("changed\n")),
        approval_callback=cb,
    )
    # No automatic approval: every mutating action must ask the Core.
    assert any(e["event_type"] == EventType.APPROVAL_REQUIRED.value for e in result.events)
    # Denied -> the write must never have happened.
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "orig\n"
    assert any(e["event_type"] == EventType.APPROVAL_DENIED.value for e in result.events)
    assert result.ok is False


def test_events_ordered_and_completed(tmp_path: Path):
    (tmp_path / "README.md").write_text("orig\n", encoding="utf-8")
    result = run_task(
        "حدّث README",
        root=tmp_path,
        approval_mode=ApprovalMode.CONFIRM,
        llm_client=FakeLLMClient(tmp_path, plan_override=_write_plan("new\n")),
        approval_callback=lambda r: True,
    )
    types = [e["event_type"] for e in result.events]
    assert types[0] == EventType.TASK_ACCEPTED.value
    assert types.index(EventType.INTENT_CLASSIFIED.value) < types.index(EventType.SNAPSHOT_READY.value)
    assert types.index(EventType.SNAPSHOT_READY.value) < types.index(EventType.PLAN_READY.value)
    assert types.index(EventType.PLAN_READY.value) < types.index(EventType.VERIFICATION_STARTED.value)
    assert types.index(EventType.VERIFICATION_STARTED.value) < types.index(EventType.VERIFICATION_PASSED.value)
    assert types.index(EventType.VERIFICATION_PASSED.value) < types.index(EventType.TASK_COMPLETED.value)
    # sequence numbers strictly increasing
    seqs = [e["seq"] for e in result.events]
    assert seqs == sorted(seqs)
    assert result.ok is True


def test_approval_required_then_accepted(tmp_path: Path):
    (tmp_path / "README.md").write_text("orig\n", encoding="utf-8")
    seen = []

    def cb(request):
        seen.append(request)
        return True

    result = run_task(
        "عدّل README",
        root=tmp_path,
        approval_mode=ApprovalMode.CONFIRM,
        llm_client=FakeLLMClient(tmp_path, plan_override=_write_plan("changed\n")),
        approval_callback=cb,
    )
    types = [e["event_type"] for e in result.events]
    assert EventType.APPROVAL_REQUIRED.value in types
    assert EventType.APPROVAL_ACCEPTED.value in types
    assert "changed" in (tmp_path / "README.md").read_text(encoding="utf-8")
    assert result.ok is True


def test_verification_failed_rollback_event(tmp_path: Path):
    (tmp_path / "README.md").write_text("orig\n", encoding="utf-8")
    plan = {
        "summary": "ui4 failing verify",
        "steps": ["write", "verify"],
        "actions": [
            {"name": "read_file", "arguments": {"path": "README.md"}},
            {"name": "write_file", "arguments": {"path": "README.md", "content": "x\n"}},
        ],
        "verification": [FAIL_CMD],
    }
    result = run_task(
        "عدّل README",
        root=tmp_path,
        approval_mode=ApprovalMode.CONFIRM,
        max_rounds=3,
        llm_client=FakeLLMClient(tmp_path, plan_override=plan),
        approval_callback=lambda r: True,
    )
    types = [e["event_type"] for e in result.events]
    assert EventType.VERIFICATION_FAILED.value in types
    assert EventType.ROLLBACK_COMPLETED.value in types
    assert result.ok is False


def test_unknown_change_detected(tmp_path: Path):
    import nabd.agent as agent_mod

    (tmp_path / "README.md").write_text("orig\n", encoding="utf-8")
    orig_snap = agent_mod.take_snapshot
    calls = {"n": 0}

    def fake_snap(path):
        calls["n"] += 1
        snap = orig_snap(path)
        if calls["n"] > 1:  # inject an external change after the before-snapshot
            snap = dict(snap)
            snap["injected_unknown_file.txt"] = "z"
        return snap

    agent_mod.take_snapshot = fake_snap
    try:
        result = run_task(
            "اقرأ README",
            root=tmp_path,
            approval_mode=ApprovalMode.CONFIRM,
            llm_client=FakeLLMClient(
                tmp_path,
                plan_override={
                    "summary": "ro",
                    "steps": ["read", "verify"],
                    "actions": [{"name": "read_file", "arguments": {"path": "README.md"}}],
                    "verification": [PASS_CMD],
                },
            ),
            approval_callback=lambda r: True,
        )
    finally:
        agent_mod.take_snapshot = orig_snap
    types = [e["event_type"] for e in result.events]
    assert EventType.UNKNOWN_CHANGE_DETECTED.value in types
    assert "injected_unknown_file.txt" in str(result.events)


def test_timeout_event(tmp_path: Path):
    import nabd.agent as agent_mod
    import time as _time

    real = _time.time
    counter = {"n": 0}

    def fake_time():
        counter["n"] += 1
        # First call seeds start_time in the past; later calls are "now".
        return real() - 400 if counter["n"] == 1 else real()

    agent_mod.time.time = fake_time
    try:
        result = run_task(
            "اقرأ README",
            root=tmp_path,
            approval_mode=ApprovalMode.CONFIRM,
            llm_client=FakeLLMClient(
                tmp_path,
                plan_override={
                    "summary": "ro",
                    "steps": ["read"],
                    "actions": [{"name": "read_file", "arguments": {"path": "README.md"}}],
                },
            ),
            approval_callback=lambda r: True,
        )
    finally:
        agent_mod.time.time = real
    types = [e["event_type"] for e in result.events]
    assert EventType.TIMEOUT.value in types
    assert result.ok is False


def test_task_completed_only_with_evidence(tmp_path: Path):
    (tmp_path / "README.md").write_text("orig\n", encoding="utf-8")
    result = run_task(
        "عدّل README",
        root=tmp_path,
        approval_mode=ApprovalMode.CONFIRM,
        llm_client=FakeLLMClient(tmp_path, plan_override=_write_plan("new\n")),
        approval_callback=lambda r: True,
    )
    assert result.ok is True
    types = [e["event_type"] for e in result.events]
    assert EventType.TASK_COMPLETED.value in types
    # completed only when there is at least one valid OBSERVED evidence record
    assert any(e.get("type") == "OBSERVED" for e in result.evidence)


def test_fake_client_no_network(tmp_path: Path):
    (tmp_path / "README.md").write_text("orig\n", encoding="utf-8")
    client = FakeLLMClient(tmp_path, plan_override=_write_plan("new\n"))
    run_task(
        "عدّل README",
        root=tmp_path,
        approval_mode=ApprovalMode.CONFIRM,
        llm_client=client,
        approval_callback=lambda r: True,
    )
    assert client.calls  # planning happened offline, no real provider


def test_run_demo_not_in_runtime():
    import nabd.runtime as rt

    # run_demo must stay in UI-3 standalone; Runtime exposes run_task only.
    assert not hasattr(rt, "run_demo"), "run_demo must not be exposed by Runtime"


def test_ui_crash_does_not_break_core(tmp_path: Path):
    (tmp_path / "README.md").write_text("orig\n", encoding="utf-8")

    class BrokenSink:
        def publish(self, event):
            raise RuntimeError("renderer exploded")

    result = run_task(
        "عدّل README",
        root=tmp_path,
        approval_mode=ApprovalMode.CONFIRM,
        llm_client=FakeLLMClient(tmp_path, plan_override=_write_plan("new\n")),
        event_sink=BrokenSink(),
        approval_callback=lambda r: True,
    )
    # A broken UI sink must never crash the agent or bypass a safety policy.
    assert result.ok is True
