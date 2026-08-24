"""Data contracts exchanged between the planner, tools, and verifier."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    summary: str
    steps: List[str]
    tool_calls: List[ToolCall] = field(default_factory=list)
    verification: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    name: str
    ok: bool
    output: str
    exit_code: Optional[int] = None
    raw_facts: Any = None

    def as_dict(self) -> Dict[str, Any]:
        facts = self.raw_facts
        if hasattr(facts, "to_dict"):
            facts = facts.to_dict()
        return {
            "tool": self.name,
            "ok": self.ok,
            "output": self.output,
            "exit_code": self.exit_code,
            "raw_facts": facts,
        }


@dataclass
class AgentResult:
    ok: bool
    state: str
    summary: str
    changes: List[str] = field(default_factory=list)
    verification: List[ToolResult] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
