"""Generate exactly six-block problems with PlanBench's bundled IPC generator.

Run this on Linux after building the two generator helper binaries. The raw
PDDL output is kept, because the bundled `./blocksworld` wrapper has no seed
argument; the saved files, hashes, and source revision define the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .pddl import parse_problem


PLANBENCH_REVISION = "1f4d600f4806bedbefebbbe44b168372aaf5060d"
PLANBENCH_REPOSITORY = "https://github.com/harshakokel/PlanBench"
GENERATOR_RELATIVE = Path("plan-bench/pddlgenerators/blocksworld/blocksworld")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(checkout: Path, output: Path, *, count: int = 500) -> dict:
    if count < 1:
        raise ValueError("Dataset size must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty dataset directory: {output}")
    checkout = checkout.resolve()
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if revision != PLANBENCH_REVISION:
        raise ValueError(f"PlanBench revision must be {PLANBENCH_REVISION}; got {revision}")
    script = checkout / GENERATOR_RELATIVE
    state_binary = script.parent / "bwstates.1/bwstates"
    pddl_binary = script.parent / "4ops/2pddl/2pddl"
    for path in (script, state_binary, pddl_binary):
        if not path.is_file():
            raise FileNotFoundError(f"Build or restore PlanBench generator file: {path}")
    output.mkdir(parents=True, exist_ok=True)
    instances = []
    seen: set[tuple] = set()
    duplicate_streak = 0
    attempts = 0
    trivial = 0
    duplicates = 0
    while len(instances) < count:
        attempts += 1
        result = subprocess.run(
            [str(script), "4", "6"], cwd=script.parent, capture_output=True,
            text=True, check=True, timeout=30,
        )
        raw = result.stdout
        problem = parse_problem(raw, required_blocks=6)
        identity = (problem.initial, problem.goal)
        if identity in seen:
            duplicates += 1
            duplicate_streak += 1
            if duplicate_streak >= 50:
                raise RuntimeError("50 consecutive duplicate instances; dataset incomplete")
            continue
        duplicate_streak = 0
        if problem.achieved(problem.initial):
            trivial += 1
            continue
        seen.add(identity)
        filename = f"instance-{len(instances)+1:04d}.pddl"
        path = output / filename
        path.write_text(raw, encoding="utf-8", newline="\n")
        instances.append({"file": filename, "sha256": sha256(path)})
    manifest = {
        "schema_version": 1,
        "source_repository": PLANBENCH_REPOSITORY,
        "source_revision": revision,
        "generator_command": "./blocksworld 4 6",
        "generator_files": {
            str(path.relative_to(checkout)).replace("\\", "/"): sha256(path)
            for path in (script, state_binary, pddl_binary)
        },
        "block_count": 6,
        "instance_count": len(instances),
        "candidate_calls": attempts,
        "rejected_trivial_initial_goal": trivial,
        "rejected_duplicate_initial_goal": duplicates,
        "length_filter": None,
        "instances": instances,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planbench-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()
    result = generate(args.planbench_checkout, args.output, count=args.count)
    print(f"Generated {result['instance_count']} six-block instances in {args.output}")


if __name__ == "__main__":
    main()
