import unittest

from nabd.fsm import FSM, FSMError, State


class FSMTests(unittest.TestCase):
    def test_happy_path_requires_verification(self):
        fsm = FSM()
        fsm.transition(State.EXECUTING)
        fsm.transition(State.VERIFYING)
        fsm.transition(State.COMPLETED)
        self.assertTrue(fsm.is_terminal())
        self.assertEqual(
            [pair[1] for pair in fsm.history],
            [State.EXECUTING, State.VERIFYING, State.COMPLETED],
        )

    def test_planning_cannot_skip_verification(self):
        fsm = FSM()
        with self.assertRaises(FSMError):
            fsm.transition(State.COMPLETED)

    def test_verification_can_return_to_execution(self):
        fsm = FSM()
        fsm.transition(State.EXECUTING)
        fsm.transition(State.VERIFYING)
        fsm.transition(State.EXECUTING)
        self.assertIs(fsm.state, State.EXECUTING)

    def test_complete_requires_external_verification(self):
        fsm = FSM()
        fsm.transition(State.EXECUTING)
        fsm.transition(State.VERIFYING)
        with self.assertRaises(FSMError):
            fsm.complete(False)
        fsm.complete(True)
        self.assertEqual(fsm.state, State.COMPLETED)

    def test_rejected_is_terminal(self):
        fsm = FSM()
        fsm.transition(State.REJECTED)
        self.assertTrue(fsm.is_terminal())
        with self.assertRaises(FSMError):
            fsm.transition(State.EXECUTING)

    def test_allowed_next_and_can_transition(self):
        fsm = FSM()
        self.assertTrue(fsm.can_transition(State.EXECUTING))
        self.assertFalse(fsm.can_transition(State.COMPLETED))
        self.assertEqual(fsm.allowed_next(), {State.EXECUTING, State.REJECTED})


if __name__ == "__main__":
    unittest.main()
