"""Action-mean advantages and the confirmed two-step gate.

All statistics are computed from a single frozen rollout batch. The Game24
INVALID action participates in action-mean statistics but cannot pass the
five-legal-action gate or enter the selected second-action set.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
from math import ceil, sqrt
from random import Random
from typing import Iterable

from .env import INVALID


@dataclass(frozen=True)
class Visit:
    visit_id: str
    trajectory_id: str
    step_index: int
    state: tuple[str, ...]
    action: str
    child_state: tuple[str, ...] | None
    reward: int


@dataclass
class StateStats:
    state: tuple[str, ...]
    visits: list[str]
    action_visits: dict[str, list[str]]
    q: dict[str, Fraction]
    mean: Fraction
    variance: Fraction
    sigma: float
    advantage: dict[str, float]
    observed_legal_actions: int

    def record(self) -> dict:
        return {
            "state": self.state,
            "visits": self.visits,
            "action_visits": self.action_visits,
            "q": {key: str(value) for key, value in self.q.items()},
            "mean": str(self.mean),
            "variance": str(self.variance),
            "sigma": self.sigma,
            "advantage": self.advantage,
            "observed_legal_actions": self.observed_legal_actions,
            "statistical_actions": len(self.q),
        }


@dataclass
class Gate:
    parent_state: tuple[str, ...]
    parent_action: str
    child_state: tuple[str, ...] | None
    parent_sigma: float
    child_sigma: float | None
    observed_legal_actions: int
    passes_five: bool
    passes_delta: bool
    passed: bool
    selected_second_actions: list[str]
    selection_count: int
    reason: str

    def record(self) -> dict:
        record = asdict(self)
        record["sigma_ratio"] = (
            self.child_sigma / self.parent_sigma
            if self.child_sigma is not None and self.parent_sigma > 0 else None
        )
        return record


@dataclass
class VisitAdvantage:
    visit_id: str
    base: float
    selected_first: bool
    parent_gate_passed: bool
    B: bool
    P: bool
    selected_second: bool
    overlap: bool
    lambda_contribution: float
    beta_contribution: float
    final: float

    def record(self) -> dict:
        return asdict(self)


@dataclass
class AdvantageBatch:
    states: dict[tuple[str, ...], StateStats]
    selected_first_visits: set[str]
    gates: dict[tuple[tuple[str, ...], str], Gate]
    advantages: dict[str, VisitAdvantage]


def _select_with_boundary_ties(
    keys: list[str], scores: dict[str, Fraction], count: int,
    largest: bool, rng: Random,
) -> list[str]:
    if count <= 0:
        return []
    if count > len(keys):
        raise ValueError("Selection count exceeds candidates")
    ordered = sorted((scores[key] for key in keys), reverse=largest)
    boundary = ordered[count - 1]
    if largest:
        strict = [key for key in keys if scores[key] > boundary]
    else:
        strict = [key for key in keys if scores[key] < boundary]
    tied = sorted(key for key in keys if scores[key] == boundary)
    chosen = strict + rng.sample(tied, count - len(strict))
    return chosen


def _combine(base: float, B: bool, P: bool, selected_second: bool,
             lambda_bonus: float, beta: float) -> tuple[float, float, float]:
    """Return final advantage, lambda contribution, second-action contribution.

    The overlap case deliberately uses base + lambda + beta*A2, matching
    section 4.3 of the existing theoretical document. Here A2 == base.
    """
    first = lambda_bonus if B else 0.0
    if P and selected_second and B:
        second = beta * base
    elif P and selected_second:
        second = (beta - 1.0) * base
    else:
        second = 0.0
    return base + first + second, first, second


def compute_advantages(
    visits: Iterable[Visit], *, omega: float, lambda_bonus: float,
    beta: float, seed: int,
) -> AdvantageBatch:
    if omega < 0 or lambda_bonus <= 0 or beta <= 1:
        raise ValueError("Require omega >= 0, lambda > 0, beta > 1")
    rows = list(visits)
    ids = [row.visit_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Visit IDs must be unique")
    by_state: dict[tuple[str, ...], list[Visit]] = defaultdict(list)
    by_trajectory: dict[str, list[Visit]] = defaultdict(list)
    for row in rows:
        if row.reward not in (0, 1):
            raise ValueError("Terminal reward must be binary")
        by_state[row.state].append(row)
        by_trajectory[row.trajectory_id].append(row)

    stats: dict[tuple[str, ...], StateStats] = {}
    for state in sorted(by_state):
        grouped: dict[str, list[Visit]] = defaultdict(list)
        for row in by_state[state]:
            grouped[row.action].append(row)
        q = {
            action: Fraction(sum(row.reward for row in action_rows), len(action_rows))
            for action, action_rows in grouped.items()
        }
        mean = sum(q.values(), Fraction(0)) / len(q)
        variance = sum(((value - mean) ** 2 for value in q.values()), Fraction(0)) / len(q)
        sigma = sqrt(float(variance))
        advantage = {
            action: (float((value - mean) / sigma) if sigma > 0 else 0.0)
            for action, value in q.items()
        }
        stats[state] = StateStats(
            state=state,
            visits=[row.visit_id for row in by_state[state]],
            action_visits={action: [row.visit_id for row in action_rows]
                           for action, action_rows in grouped.items()},
            q=q, mean=mean, variance=variance, sigma=sigma, advantage=advantage,
            observed_legal_actions=sum(action != INVALID for action in q),
        )

    rng = Random(seed)
    first_selected: set[str] = set()
    for state in sorted(stats):
        info = stats[state]
        if info.sigma == 0:
            continue
        state_rows = by_state[state]
        scores = {row.visit_id: abs(info.q[row.action] - info.mean) for row in state_rows}
        selected = _select_with_boundary_ties(
            [row.visit_id for row in state_rows], scores,
            ceil(len(state_rows) / 5), False, rng,
        )
        first_selected.update(selected)

    gates: dict[tuple[tuple[str, ...], str], Gate] = {}
    for state in sorted(stats):
        info = stats[state]
        candidate_actions = sorted({row.action for row in by_state[state]
                                    if row.visit_id in first_selected})
        for action in candidate_actions:
            matching = [row for row in by_state[state] if row.action == action]
            children = {row.child_state for row in matching}
            if len(children) != 1:
                raise ValueError(f"Inconsistent children for {state}, {action}")
            child_state = next(iter(children))
            child_info = stats.get(child_state) if child_state is not None else None
            d_valid = child_info.observed_legal_actions if child_info else 0
            child_sigma = child_info.sigma if child_info else None
            five = d_valid >= 5
            delta = bool(child_info and child_info.variance > 0
                         and child_info.variance >= Fraction(str(omega)) ** 2 * info.variance)
            passed = bool(action != INVALID and child_info and five and delta)
            selected_second: list[str] = []
            selection_count = 0
            if passed:
                assert child_info is not None
                legal = sorted(key for key in child_info.q if key != INVALID)
                selection_count = ceil(len(legal) / 2)
                scores = {key: abs(child_info.q[key] - child_info.mean) for key in legal}
                selected_second = _select_with_boundary_ties(
                    legal, scores, selection_count, True, rng,
                )
            reason = "passed" if passed else (
                "invalid_parent" if action == INVALID else
                "no_child_visits" if child_info is None else
                "fewer_than_five_legal_actions" if not five else
                "zero_child_sigma" if child_sigma == 0 else "delta_gate"
            )
            gates[(state, action)] = Gate(
                state, action, child_state, info.sigma, child_sigma,
                d_valid, five, delta, passed, selected_second,
                selection_count, reason,
            )

    previous: dict[str, Visit] = {}
    for trajectory_id, trajectory_rows in by_trajectory.items():
        ordered = sorted(trajectory_rows, key=lambda row: row.step_index)
        if len({row.step_index for row in ordered}) != len(ordered):
            raise ValueError(f"Repeated step index in {trajectory_id}")
        for prior, current in zip(ordered, ordered[1:]):
            if current.step_index != prior.step_index + 1:
                raise ValueError(f"Non-adjacent visits in {trajectory_id}")
            previous[current.visit_id] = prior

    outputs: dict[str, VisitAdvantage] = {}
    for row in rows:
        base = stats[row.state].advantage[row.action]
        gate = gates.get((row.state, row.action))
        first = row.visit_id in first_selected
        gate_passed = bool(gate and gate.passed)
        B = first and gate_passed
        prior = previous.get(row.visit_id)
        prior_gate = gates.get((prior.state, prior.action)) if prior else None
        P = bool(prior_gate and prior_gate.passed)
        second = bool(P and prior_gate and row.action in prior_gate.selected_second_actions)
        final, first_term, second_term = _combine(
            base, B, P, second, lambda_bonus, beta,
        )
        outputs[row.visit_id] = VisitAdvantage(
            row.visit_id, base, first, gate_passed, B, P, second,
            B and P, first_term, second_term, final,
        )
    return AdvantageBatch(stats, first_selected, gates, outputs)
