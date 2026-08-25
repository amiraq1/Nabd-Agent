"""Finite State Machine for Nabd Agent V3.

States:
  PLANNING -> EXECUTING -> VERIFYING -> COMPLETED
                                  |-> REPAIRING -> EXECUTING (loop)
                                  |-> ROLLED_BACK (terminal)
                                  |-> FAILED (terminal)
Terminal states: COMPLETED, REJECTED, ROLLED_BACK, FAILED.

Planning cannot skip verification.  REPAIRING allows bounded retry
loops.  ROLLED_BACK and FAILED are absorbing states that prevent any
further tool calls.
"""

from enum import Enum, auto
from typing import Dict, List, Set, Tuple


class State(Enum):
    PLANNING = auto()
    EXECUTING = auto()
    VERIFYING = auto()
    COMPLETED = auto()
    REJECTED = auto()
    REPAIRING = auto()
    ROLLED_BACK = auto()
    FAILED = auto()


ALLOWED_TRANSITIONS: Dict[State, Set[State]] = {
    State.PLANNING: {State.EXECUTING, State.REJECTED},
    State.EXECUTING: {State.VERIFYING, State.REJECTED},
    State.VERIFYING: {
        State.COMPLETED,
        State.EXECUTING,       # implicit repair loop (legacy)
        State.REPAIRING,       # explicit repair with budget tracking
        State.ROLLED_BACK,     # irreparable failure
        State.FAILED,          # timeout or blocked
        State.REJECTED,
    },
    State.REPAIRING: {State.EXECUTING, State.ROLLED_BACK, State.REJECTED},
    State.COMPLETED: set(),
    State.REJECTED: set(),
    State.ROLLED_BACK: set(),
    State.FAILED: set(),
}


class FSMError(Exception):
    """Raised when an invalid state transition is attempted."""


class FSM:
    """Minimal FSM with history tracking and strict transition validation."""

    def __init__(self, initial_state: State = State.PLANNING):
        self.state = initial_state
        self.history: List[Tuple[State, State]] = []

    def transition(self, target: State) -> None:
        """Transition to target state if allowed, otherwise raise FSMError."""
        if target not in ALLOWED_TRANSITIONS[self.state]:
            allowed = [state.name for state in ALLOWED_TRANSITIONS[self.state]]
            raise FSMError(
                f"Invalid transition: {self.state.name} -> {target.name}. "
                f"Allowed: {allowed}"
            )
        self.history.append((self.state, target))
        self.state = target

    def complete(self, verified: bool) -> None:
        """Enter COMPLETED only when the external verifier returned true."""
        if not verified:
            raise FSMError("Cannot complete without usable verified evidence")
        self.transition(State.COMPLETED)

    def is_terminal(self) -> bool:
        return self.state in (
            State.COMPLETED,
            State.REJECTED,
            State.ROLLED_BACK,
            State.FAILED,
        )

    def allowed_next(self) -> Set[State]:
        return set(ALLOWED_TRANSITIONS[self.state])

    def can_transition(self, target: State) -> bool:
        return target in ALLOWED_TRANSITIONS[self.state]
