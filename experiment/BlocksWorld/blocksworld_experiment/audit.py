"""Read-only audit of PlanBench-generated six-block PDDL instances."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .domain import shortest_plan
from .pddl import parse_problem


def audit(directory: Path) -> dict:
    paths = sorted(directory.glob("*.pddl"))
    if not paths:
        raise ValueError(f"No PDDL problem files in {directory}")
    length_counts: Counter[int] = Counter()
    initial_branches: Counter[int] = Counter()
    optimal_path_branches: Counter[int] = Counter()
    source_files = []
    seen = set()
    for path in paths:
        raw = path.read_bytes()
        problem = parse_problem(raw.decode("utf-8"), required_blocks=6)
        identity = (problem.initial, problem.goal)
        if identity in seen:
            raise ValueError(f"Duplicate initial state and goal in {path}")
        seen.add(identity)
        plan = shortest_plan(problem)
        length_counts[len(plan)] += 1
        initial_branches[len(problem.initial.legal_actions())] += 1
        state = problem.initial
        for action in plan:
            optimal_path_branches[len(state.legal_actions())] += 1
            state = state.apply(action)
        assert problem.achieved(state)
        source_files.append({"file": path.name,
                             "sha256": hashlib.sha256(raw).hexdigest(),
                             "shortest_plan_length": len(plan),
                             "initial_legal_actions": len(problem.initial.legal_actions())})
    return {
        "block_count": 6,
        "instance_count": len(paths),
        "shortest_plan_length_counts": dict(sorted(length_counts.items())),
        "exceeds_16_step_limit": sum(count for length, count in length_counts.items()
                                     if length > 16),
        "initial_legal_action_counts": dict(sorted(initial_branches.items())),
        "one_shortest_plan_decision_legal_action_counts":
            dict(sorted(optimal_path_branches.items())),
        "note": "Optimal-path branch counts use one BFS tie-break path per instance; "
                "they are not rollout observations or the five-observed-action gate rate.",
        "sources": source_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
