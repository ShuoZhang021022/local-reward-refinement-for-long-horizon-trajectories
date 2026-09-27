"""Exact symbolic transitions for PlanBench's Blocksworld-4ops domain.

Stacks are stored bottom-to-top. Their order on the table is immaterial.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Atom:
    predicate: str
    arguments: tuple[str, ...] = ()


@dataclass(frozen=True, order=True)
class Action:
    name: str
    block: str
    support: str | None = None

    def __post_init__(self) -> None:
        if self.name not in {"pick-up", "put-down", "stack", "unstack"}:
            raise ValueError(f"Unknown Blocksworld action: {self.name}")
        if (self.support is None) != (self.name in {"pick-up", "put-down"}):
            raise ValueError(f"Wrong argument count for {self.name}")
        if self.support == self.block:
            raise ValueError("A block cannot support itself")

    @property
    def key(self) -> str:
        args = self.block if self.support is None else f"{self.block},{self.support}"
        return f"{self.name}({args})"


@dataclass(frozen=True)
class State:
    stacks: tuple[tuple[str, ...], ...]
    holding: str | None = None

    def __post_init__(self) -> None:
        if any(not stack for stack in self.stacks):
            raise ValueError("Empty stacks are not physical states")
        blocks = [block for stack in self.stacks for block in stack]
        if self.holding is not None:
            blocks.append(self.holding)
        if len(blocks) != len(set(blocks)) or not blocks or any(not b for b in blocks):
            raise ValueError("Blocks must have distinct nonempty names")
        object.__setattr__(self, "stacks", tuple(sorted(self.stacks)))

    @property
    def blocks(self) -> tuple[str, ...]:
        return tuple(sorted({b for stack in self.stacks for b in stack}
                            | ({self.holding} if self.holding is not None else set())))

    @property
    def atoms(self) -> frozenset[Atom]:
        facts = set()
        for stack in self.stacks:
            facts.add(Atom("ontable", (stack[0],)))
            facts.add(Atom("clear", (stack[-1],)))
            for lower, upper in zip(stack, stack[1:]):
                facts.add(Atom("on", (upper, lower)))
        if self.holding is None:
            facts.add(Atom("handempty"))
        else:
            facts.add(Atom("holding", (self.holding,)))
        return frozenset(facts)

    def legal_actions(self) -> tuple[Action, ...]:
        if self.holding is not None:
            actions = [Action("put-down", self.holding)]
            actions.extend(Action("stack", self.holding, stack[-1])
                           for stack in self.stacks)
        else:
            actions = []
            for stack in self.stacks:
                if len(stack) == 1:
                    actions.append(Action("pick-up", stack[-1]))
                else:
                    actions.append(Action("unstack", stack[-1], stack[-2]))
        return tuple(sorted(actions))

    def apply(self, action: Action) -> State:
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action.key} in {self}")
        if action.name == "put-down":
            assert self.holding is not None
            return State(self.stacks + ((self.holding,),))
        if action.name == "stack":
            assert self.holding is not None
            stacks = tuple(stack + (self.holding,) if stack[-1] == action.support else stack
                           for stack in self.stacks)
            return State(stacks)
        if action.name == "pick-up":
            stacks = tuple(stack for stack in self.stacks if stack[-1] != action.block)
        else:
            stacks = tuple(stack[:-1] if stack[-1] == action.block else stack
                           for stack in self.stacks)
        return State(stacks, action.block)


@dataclass(frozen=True)
class Problem:
    blocks: tuple[str, ...]
    initial: State
    goal: tuple[Atom, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(self.blocks)) != self.initial.blocks:
            raise ValueError("The object set and initial state disagree")
        if not self.goal:
            raise ValueError("A problem needs a nonempty goal")
        if any(arg not in self.blocks for atom in self.goal for arg in atom.arguments):
            raise ValueError("Goal refers to an unknown block")

    def achieved(self, state: State) -> bool:
        return state.blocks == self.blocks and set(self.goal) <= state.atoms

    def decision_key(self, state: State, *, steps_remaining: int) -> tuple:
        """The full decision state includes layout, goal, and action budget."""
        if steps_remaining < 0:
            raise ValueError("Remaining action budget cannot be negative")
        return (state.stacks, state.holding, steps_remaining,
                tuple(sorted(self.goal)))


def shortest_plan(problem: Problem) -> tuple[Action, ...]:
    """Breadth-first search over unit-cost actions; intended for six blocks."""
    start = problem.initial
    if problem.achieved(start):
        return ()
    queue = deque([start])
    parents: dict[State, tuple[State, Action] | None] = {start: None}
    while queue:
        current = queue.popleft()
        for action in current.legal_actions():
            successor = current.apply(action)
            if successor in parents:
                continue
            parents[successor] = (current, action)
            if problem.achieved(successor):
                plan = []
                at = successor
                while parents[at] is not None:
                    previous, chosen = parents[at]
                    plan.append(chosen)
                    at = previous
                return tuple(reversed(plan))
            queue.append(successor)
    raise ValueError("No plan achieves the goal")
