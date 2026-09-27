"""Check the local transition system against PlanBench's four PDDL operators."""

import unittest

from blocksworld_experiment.domain import Action, Atom, Problem, State, shortest_plan
from blocksworld_experiment.env import Blocksworld, INVALID, parse_action
from blocksworld_experiment.pddl import parse_problem


OFFICIAL_STYLE_PROBLEM = """
(define (problem BW-rand-4)
  (:domain blocksworld-4ops)
  (:objects a b c d)
  (:init (handempty) (ontable a) (on b c) (ontable c)
         (ontable d) (clear a) (clear b) (clear d))
  (:goal (and (on c b)))
)
"""


class DomainTests(unittest.TestCase):
    def test_official_style_pddl_and_four_operator_effects(self) -> None:
        problem = parse_problem(OFFICIAL_STYLE_PROBLEM, required_blocks=4)
        start = problem.initial
        self.assertEqual(start.stacks, (("a",), ("c", "b"), ("d",)))
        self.assertEqual({a.key for a in start.legal_actions()},
                         {"pick-up(a)", "unstack(b,c)", "pick-up(d)"})
        holding = start.apply(Action("unstack", "b", "c"))
        self.assertIn(Atom("clear", ("c",)), holding.atoms)
        self.assertNotIn(Atom("handempty"), holding.atoms)
        self.assertEqual({a.key for a in holding.legal_actions()},
                         {"put-down(b)", "stack(b,a)", "stack(b,c)", "stack(b,d)"})
        self.assertEqual(holding.apply(Action("stack", "b", "c")), start)
        with self.assertRaises(ValueError):
            start.apply(Action("pick-up", "c"))

    def test_shortest_plan_and_goal_in_decision_key(self) -> None:
        start = State((("b", "a"), ("c",), ("d",), ("e",), ("f",)))
        problem = Problem(start.blocks, start, (Atom("on", ("a", "c")),))
        plan = shortest_plan(problem)
        self.assertEqual(len(plan), 2)
        self.assertEqual([action.key for action in plan],
                         ["unstack(a,b)", "stack(a,c)"])
        target = start
        for action in plan:
            target = target.apply(action)
        self.assertTrue(problem.achieved(target))
        other_goal = Problem(start.blocks, start, (Atom("on", ("a", "d")),))
        self.assertNotEqual(problem.decision_key(start, steps_remaining=16),
                            other_goal.decision_key(start, steps_remaining=16))
        self.assertNotEqual(problem.decision_key(start, steps_remaining=16),
                            problem.decision_key(start, steps_remaining=15))

    def test_rejects_incomplete_or_wrong_size_problem(self) -> None:
        with self.assertRaises(ValueError):
            parse_problem(OFFICIAL_STYLE_PROBLEM, required_blocks=6)
        with self.assertRaises(ValueError):
            parse_problem(OFFICIAL_STYLE_PROBLEM.replace("(clear b)", ""))

    def test_interactive_success_and_invalid_action(self) -> None:
        start = State((("b", "a"), ("c",), ("d",), ("e",), ("f",)))
        problem = Problem(start.blocks, start, (Atom("on", ("a", "c")),))
        game = Blocksworld(problem)
        first = game.step("unstack a b")
        self.assertEqual(first.legal_action_count, 5)
        self.assertEqual(first.action_key, "unstack(a,b)")
        self.assertFalse(first.terminal)
        last = game.step("stack a c")
        self.assertTrue(last.terminal)
        self.assertEqual(last.reward, 1)
        self.assertEqual(game.termination, "success")
        self.assertEqual(game.steps, 2)

        invalid = Blocksworld(problem).step("pick-up b")
        self.assertEqual(invalid.action_key, INVALID)
        self.assertEqual(invalid.error, "precondition")
        self.assertEqual(invalid.reward, 0)
        self.assertIsNone(invalid.after)
        self.assertEqual(parse_action("stack a c extra")[1], "format")

    def test_step_cap_is_terminal_failure(self) -> None:
        start = State((("b", "a"), ("c",), ("d",), ("e",), ("f",)))
        problem = Problem(start.blocks, start, (Atom("on", ("a", "c")),))
        game = Blocksworld(problem, max_steps=1)
        result = game.step("unstack a b")
        self.assertEqual(result.reward, 0)
        self.assertEqual(result.termination, "step_limit")


if __name__ == "__main__":
    unittest.main()
