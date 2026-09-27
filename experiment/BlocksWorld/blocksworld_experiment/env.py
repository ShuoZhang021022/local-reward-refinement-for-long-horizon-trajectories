"""Six-block, one-action-per-call interaction under the confirmed task rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from .domain import Action, Atom, Problem, State


INVALID = "INVALID"
ACTION_PATTERN = re.compile(r"\s*(pick-up|put-down|stack|unstack)\s+([a-z][a-z0-9_-]*)(?:\s+([a-z][a-z0-9_-]*))?\s*", re.I)


def atom_text(atom: Atom) -> str:
    return f"{atom.predicate}({','.join(atom.arguments)})"


def decision_key(problem: Problem, state: State, *, steps_remaining: int) -> tuple[str, ...]:
    return (f"steps_remaining={steps_remaining}",) + \
        tuple(sorted(map(atom_text, state.atoms))) + ("GOAL",) + \
        tuple(sorted(map(atom_text, problem.goal)))


def parse_action(raw: str) -> tuple[Action | None, str | None]:
    match = ACTION_PATTERN.fullmatch(raw)
    if match is None:
        return None, "format"
    try:
        return Action(match[1].lower(), match[2].lower(),
                      match[3].lower() if match[3] else None), None
    except ValueError:
        return None, "argument_count"


@dataclass(frozen=True)
class Transition:
    before: tuple[str, ...]
    after: tuple[str, ...] | None
    raw_output: str
    action: Action | None
    action_key: str
    error: str | None
    terminal: bool
    reward: int | None
    termination: str | None
    legal_action_count: int

    def record(self) -> dict:
        return asdict(self)


class Blocksworld:
    def __init__(self, problem: Problem, *, max_steps: int = 16) -> None:
        if len(problem.blocks) != 6:
            raise ValueError("This experiment uses exactly six blocks")
        if max_steps < 1:
            raise ValueError("Step limit must be positive")
        if problem.achieved(problem.initial):
            raise ValueError("Trivial initial-goal instances are excluded by PlanBench")
        self.problem = problem
        self.state = problem.initial
        self.max_steps = max_steps
        self.steps = 0
        self.terminal = False
        self.reward: int | None = None
        self.termination: str | None = None

    @property
    def state_key(self) -> tuple[str, ...]:
        return decision_key(self.problem, self.state,
                            steps_remaining=self.max_steps-self.steps)

    @property
    def observation(self) -> str:
        facts = ", ".join(sorted(map(atom_text, self.state.atoms)))
        goal = ", ".join(sorted(map(atom_text, self.problem.goal)))
        return (f"Current state: {facts}\n"
                f"Actions remaining: {self.max_steps-self.steps}\nGoal: {goal}")

    def fail(self, raw_output: str, error: str, action: Action | None = None) -> Transition:
        if self.terminal:
            raise RuntimeError("The trajectory has already ended")
        self.terminal = True
        self.reward = 0
        self.termination = f"invalid:{error}"
        return Transition(self.state_key, None, raw_output, action, INVALID,
                          error, True, 0, self.termination,
                          len(self.state.legal_actions()))

    def step(self, raw_output: str) -> Transition:
        if self.terminal:
            raise RuntimeError("The trajectory has already ended")
        before = self.state_key
        legal_count = len(self.state.legal_actions())
        action, error = parse_action(raw_output)
        if error is not None:
            return self.fail(raw_output, error)
        assert action is not None
        if action not in self.state.legal_actions():
            return self.fail(raw_output, "precondition", action)
        self.state = self.state.apply(action)
        self.steps += 1
        achieved = self.problem.achieved(self.state)
        self.terminal = achieved or self.steps >= self.max_steps
        self.reward = int(achieved) if self.terminal else None
        self.termination = ("success" if achieved else
                            "step_limit" if self.terminal else None)
        return Transition(before, self.state_key, raw_output, action,
                          action.key, None, self.terminal, self.reward,
                          self.termination, legal_count)
