"""Exact Game of 24 environment; the policy only observes current numbers."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re
from typing import Iterable


ACTION_PATTERN = re.compile(r"[ \t]*(\d+)[ \t]+(\d+)[ \t]+([+*/-])[ \t]*")
INVALID = "INVALID"


def fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True)
class Term:
    value: Fraction
    expression: str


@dataclass(frozen=True)
class Action:
    left: int  # 1-based index in the currently displayed list
    right: int
    op: str

    @property
    def key(self) -> str:
        left, right = self.left, self.right
        if self.op in "+*" and right < left:
            left, right = right, left
        return f"{self.op}:{left}:{right}"


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
    result: str | None
    expression: str | None


def parse_action(raw: str, count: int) -> tuple[Action | None, str | None]:
    match = ACTION_PATTERN.fullmatch(raw.strip())
    if match is None:
        return None, "format"
    left, right, op = int(match[1]), int(match[2]), match[3]
    if left < 1 or left > count or right < 1 or right > count:
        return None, "slot_out_of_range"
    if left == right:
        return None, "same_slot_twice"
    return Action(left, right, op), None


def legal_actions(values: Iterable[Fraction]) -> list[Action]:
    items = list(values)
    actions: list[Action] = []
    for left in range(1, len(items) + 1):
        for right in range(1, len(items) + 1):
            if left == right:
                continue
            if left < right:
                actions.extend((Action(left, right, "+"), Action(left, right, "*")))
            actions.append(Action(left, right, "-"))
            if items[right - 1] != 0:
                actions.append(Action(left, right, "/"))
    return actions


class Game24:
    def __init__(self, puzzle: Iterable[int]) -> None:
        numbers = tuple(int(value) for value in puzzle)
        if len(numbers) != 4:
            raise ValueError("A puzzle must contain exactly four integers")
        self.terms = [Term(Fraction(number), str(number)) for number in sorted(numbers)]
        self.steps = 0
        self.terminal = False
        self.reward: int | None = None
        self.termination: str | None = None

    @property
    def values(self) -> tuple[Fraction, ...]:
        return tuple(term.value for term in self.terms)

    @property
    def state_key(self) -> tuple[str, ...]:
        return tuple(fraction_text(value) for value in self.values)

    @property
    def observation(self) -> str:
        return "[" + ", ".join(
            f"{slot}:{fraction_text(term.value)}" for slot, term in enumerate(self.terms, start=1)
        ) + "]"

    def step(self, raw_output: str) -> Transition:
        if self.terminal:
            raise RuntimeError("The trajectory has already terminated")
        before = self.state_key
        action, error = parse_action(raw_output, len(self.terms))
        if action is not None and action.op == "/" and self.terms[action.right - 1].value == 0:
            error = "division_by_zero"
        if error is not None:
            return self.fail(raw_output, error, action)

        assert action is not None
        left_term = self.terms[action.left - 1]
        right_term = self.terms[action.right - 1]
        if action.op == "+":
            result = left_term.value + right_term.value
        elif action.op == "-":
            result = left_term.value - right_term.value
        elif action.op == "*":
            result = left_term.value * right_term.value
        else:
            result = left_term.value / right_term.value
        expression = f"({left_term.expression}{action.op}{right_term.expression})"
        self.terms = [
            term for slot, term in enumerate(self.terms, start=1)
            if slot not in (action.left, action.right)
        ]
        self.terms.append(Term(result, expression))
        self.terms.sort(key=lambda term: term.value)
        self.steps += 1
        self.terminal = self.steps == 3
        self.reward = int(self.terms[0].value == 24) if self.terminal else None
        self.termination = "success" if self.reward == 1 else ("not_24" if self.terminal else None)
        return Transition(
            before, self.state_key, raw_output, action, action.key, None,
            self.terminal, self.reward, fraction_text(result),
            self.terms[0].expression if self.terminal else None,
        )

    def fail(self, raw_output: str, error: str, action: Action | None = None) -> Transition:
        """End a trajectory for a model-generated invalid response."""
        if self.terminal:
            raise RuntimeError("The trajectory has already terminated")
        before = self.state_key
        self.steps += 1
        self.terminal = True
        self.reward = 0
        self.termination = f"invalid:{error}"
        return Transition(before, None, raw_output, action, INVALID, error, True, 0, None, None)
