"""Verify raw PlanBench problems and make an unfiltered deterministic split."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .domain import Problem, shortest_plan
from .generate import PLANBENCH_REPOSITORY, PLANBENCH_REVISION
from .pddl import parse_problem


@dataclass(frozen=True)
class Instance:
    index: int
    filename: str
    sha256: str
    problem: Problem
    shortest_length: int


def load_instances(directory: Path, *, expected_count: int) -> tuple[list[Instance], dict]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or
            manifest.get("source_repository") != PLANBENCH_REPOSITORY or
            manifest.get("source_revision") != PLANBENCH_REVISION or
            manifest.get("block_count") != 6 or
            manifest.get("instance_count") != expected_count or
            manifest.get("length_filter") is not None):
        raise ValueError("Dataset manifest does not match the six-block unfiltered protocol")
    entries = manifest.get("instances")
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise ValueError("Dataset manifest has the wrong number of entries")
    listed = {entry["file"] for entry in entries}
    existing = {path.name for path in directory.glob("*.pddl")}
    if listed != existing or len(listed) != expected_count:
        raise ValueError("Dataset file listing does not match the manifest")
    instances = []
    identities = set()
    for index, entry in enumerate(entries):
        if entry.get("file") != f"instance-{index+1:04d}.pddl":
            raise ValueError("Dataset file names/order do not match generation order")
        path = directory / entry["file"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Dataset file hash changed: {path}")
        problem = parse_problem(raw.decode("utf-8"), required_blocks=6)
        if problem.achieved(problem.initial):
            raise ValueError(f"Trivial problem in {path}")
        identity = (problem.initial, problem.goal)
        if identity in identities:
            raise ValueError(f"Duplicate initial-goal problem in {path}")
        identities.add(identity)
        instances.append(Instance(index, entry["file"], digest, problem,
                                  len(shortest_plan(problem))))
    return instances, manifest


def split_instances(instances: list[Instance], *, seed: int,
                    train_size: int, validation_size: int, test_size: int) -> dict:
    if len(instances) != train_size + validation_size + test_size:
        raise ValueError("Split sizes disagree with dataset")
    ranked = sorted(
        instances,
        key=lambda instance: hashlib.sha256(
            f"blocksworld6-split-v1:{seed}:{instance.index}:{instance.sha256}".encode()
        ).hexdigest(),
    )
    train = sorted(instance.index for instance in ranked[:train_size])
    validation = sorted(instance.index for instance in ranked[train_size:
                                                              train_size+validation_size])
    test = sorted(instance.index for instance in ranked[train_size+validation_size:])
    by_index = {instance.index: instance for instance in instances}
    return {
        "schema_version": 1,
        "method": "SHA256 rank of fixed raw instances; no optimal-length filtering or stratification",
        "seed": seed,
        "train": train,
        "validation": validation,
        "test": test,
        "shortest_length_counts": {
            part: {str(length): count for length, count in
                   sorted(Counter(by_index[i].shortest_length for i in indices).items())}
            for part, indices in (("train", train), ("validation", validation), ("test", test))
        },
        "over_16_steps": {
            part: sum(by_index[i].shortest_length > 16 for i in indices)
            for part, indices in (("train", train), ("validation", validation), ("test", test))
        },
    }
