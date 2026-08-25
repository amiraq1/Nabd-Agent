"""Verification Gate for Nabd Agent.

Prevents false-positive task completion by requiring deterministic,
evidence-backed verification of all mandatory success criteria before
allowing the FSM to transition to COMPLETED.
"""

from .gate import VerificationGate
from .types import (
    Criterion,
    CriterionKind,
    Decision,
    FailureSignature,
    Report,
    Result,
    SuccessCriteria,
)

__all__ = [
    "Criterion",
    "CriterionKind",
    "Decision",
    "FailureSignature",
    "Report",
    "Result",
    "SuccessCriteria",
    "VerificationGate",
]
