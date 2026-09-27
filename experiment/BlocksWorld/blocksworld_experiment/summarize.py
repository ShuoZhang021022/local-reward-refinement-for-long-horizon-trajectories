"""Audit completed paired Blocksworld runs and compare held-out success."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from random import Random

from game24_experiment.checkpoints import file_sha256
from game24_experiment.io import write_json

from .train import _config_from_plan


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"Missing record: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _nonnegative(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid nonnegative count for {label}: {value!r}")
    return value


def _evaluation(path: Path, expected_indices: list[int], *, max_steps: int) -> tuple[dict[int, int], dict]:
    if not path.is_file():
        raise ValueError(f"Missing evaluation trace: {path}")
    outcomes: dict[int, int] = {}
    calls = tokens = invalid = step_limit = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            index = _nonnegative(row["problem_index"], "problem_index")
            if index in outcomes:
                raise ValueError(f"Duplicate evaluation problem {index}: {path}")
            reward = row["reward"]
            if type(reward) is not int or reward not in (0, 1):
                raise ValueError(f"Invalid reward for problem {index}: {path}")
            decisions = row["decisions"]
            model_calls = _nonnegative(row["model_calls"], "model_calls")
            generated_tokens = _nonnegative(row["generated_tokens"], "generated_tokens")
            legal_executed = _nonnegative(row["legal_actions_executed"],
                                          "legal_actions_executed")
            termination = row["termination"]
            if (not 1 <= model_calls <= max_steps or row["steps"] != model_calls or
                    model_calls != len(decisions) or legal_executed > model_calls or
                    generated_tokens != sum(len(step["completion_ids"])
                                            for step in decisions) or
                    generated_tokens < model_calls or
                    (reward == 1) != (termination == "success") or
                    (termination == "step_limit" and legal_executed != max_steps)):
                raise ValueError(f"Inconsistent evaluation trajectory {index}: {path}")
            outcomes[index] = reward
            calls += model_calls
            tokens += generated_tokens
            invalid += str(termination).startswith("invalid:")
            step_limit += termination == "step_limit"
    if sorted(outcomes) != sorted(expected_indices):
        raise ValueError(f"Evaluation problem set mismatch: {path}")
    metrics = {
        "successes": sum(outcomes.values()),
        "problems": len(outcomes),
        "accuracy": sum(outcomes.values()) / len(outcomes),
        "invalid_trajectories": invalid,
        "step_limit_trajectories": step_limit,
        "model_calls": calls,
        "generated_tokens": tokens,
    }
    return outcomes, metrics


def _check_metrics(record: dict, actual: dict, path: Path) -> None:
    if any(record.get(key) != value for key, value in actual.items()):
        raise ValueError(f"Metrics disagree with evaluation trace: {path}")


def _load_run(directory: Path) -> tuple[dict, dict[int, int], dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError(f"Run is not complete: {directory}")
    plan_path = directory / "plan.json"
    if not plan_path.is_file() or file_sha256(plan_path) != manifest["plan_sha256"]:
        raise ValueError(f"Locked plan missing or changed: {directory}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan != manifest["plan"]:
        raise ValueError(f"Manifest plan disagrees with locked plan: {directory}")
    config = _config_from_plan(plan)
    if manifest["arm"] not in config.arms or manifest["seed"] not in config.training_seeds:
        raise ValueError(f"Unplanned arm or seed: {directory}")
    updates_per_epoch = ((len(plan["split"]["train"]) + config.problems_per_update - 1)
                         // config.problems_per_update)
    expected_updates = config.epochs * updates_per_epoch
    if manifest.get("completed_updates") != expected_updates:
        raise ValueError(f"Run did not finish planned training: {directory}")
    checkpoints = _read_jsonl(directory / "checkpoints.jsonl")
    if [row.get("update") for row in checkpoints] != list(range(expected_updates+1)):
        raise ValueError(f"Incomplete checkpoint list: {directory}")
    for update, record in enumerate(checkpoints):
        relative = f"checkpoints/state_{update:04d}.pt"
        snapshot = directory / relative
        if (record.get("file") != relative or
                record.get("arm") != manifest["arm"] or
                record.get("seed") != manifest["seed"] or
                record.get("plan_sha256") != manifest["plan_sha256"] or
                not snapshot.is_file() or snapshot.stat().st_size != record.get("bytes") or
                file_sha256(snapshot) != record.get("sha256")):
            raise ValueError(f"Audit checkpoint missing or changed: {snapshot}")
    order = json.loads((directory / "train_order.json").read_text(encoding="utf-8"))
    ordered = order["indices"]
    if (order.get("seed") != manifest["seed"] or
            sorted(ordered) != sorted(plan["split"]["train"])):
        raise ValueError(f"Training order disagrees with plan: {directory}")
    updates = _read_jsonl(directory / "updates.jsonl")
    if [row.get("update") for row in updates] != list(range(1, expected_updates+1)):
        raise ValueError(f"Incomplete update list: {directory}")
    counts: Counter[str] = Counter()
    mapping = {
        "training_trajectories": "trajectory_count",
        "training_generation_decisions": "decision_count",
        "training_generated_tokens": "generated_tokens",
        "training_invalid_trajectories": "invalid_trajectories",
        "observed_training_states": "state_count",
        "states_with_five_available_legal_actions":
            "states_with_five_available_legal_actions",
        "states_with_five_observed_legal_actions":
            "states_with_five_observed_legal_actions",
        "zero_sigma_states": "zero_sigma_states",
        "candidate_gate_actions": "candidate_gate_count",
        "passed_gate_actions": "passed_gate_count",
        "lambda_eligible_visits": "lambda_eligible_visit_count",
        "beta_eligible_visits": "beta_eligible_visit_count",
        "lambda_applied_visits": "lambda_visit_count",
        "beta_applied_visits": "beta_selected_visit_count",
        "overlap_applied_visits": "overlap_visit_count",
    }
    for epoch in range(config.epochs):
        if epoch:
            Random(manifest["seed"] + epoch).shuffle(ordered)
        for start in range(0, len(ordered), config.problems_per_update):
            update = epoch * updates_per_epoch + start // config.problems_per_update + 1
            row = updates[update-1]
            indices = ordered[start:start+config.problems_per_update]
            trajectories = len(indices) * config.trajectories_per_problem
            if (row.get("epoch") != epoch or row.get("problem_indices") != indices or
                    row.get("old_policy_checkpoint") != checkpoints[update-1]["file"] or
                    row.get("new_policy_checkpoint") != checkpoints[update]["file"] or
                    row.get("new_checkpoint_sha256") != checkpoints[update]["sha256"] or
                    row.get("trajectory_count") != trajectories or
                    not trajectories <= row.get("decision_count", 0) <=
                    config.max_steps * trajectories):
                raise ValueError(f"Update {update} violates the locked plan: {directory}")
            for target, source in mapping.items():
                counts[target] += _nonnegative(row[source], f"update {update} {source}")
            if (row["generated_tokens"] < row["decision_count"] or
                    row["invalid_trajectories"] > trajectories or
                    row["passed_gate_count"] > row["candidate_gate_count"]):
                raise ValueError(f"Update {update} has inconsistent counts: {directory}")
            for filename in (f"rollouts_{update:04d}.jsonl.gz",
                             f"tokens_{update:04d}.jsonl.gz",
                             f"stats_{update:04d}.json"):
                if not (directory / filename).is_file():
                    raise ValueError(f"Missing update artifact {filename}: {directory}")
    expected_validations = [0] + [
        update for update in range(1, expected_updates+1)
        if update % config.validation_every_updates == 0 or
        update % updates_per_epoch == 0
    ]
    validations = _read_jsonl(directory / "validations.jsonl")
    if [row.get("update") for row in validations] != expected_validations:
        raise ValueError(f"Incomplete validation list: {directory}")
    best_update = None
    best_accuracy = -1.0
    for update, row in zip(expected_validations, validations):
        _, actual = _evaluation(directory / f"validation_{update:04d}.jsonl.gz",
                                plan["split"]["validation"], max_steps=config.max_steps)
        _check_metrics(row, actual, directory)
        counts["validation_generation_decisions"] += actual["model_calls"]
        counts["validation_generated_tokens"] += actual["generated_tokens"]
        if actual["accuracy"] > best_accuracy:
            best_update, best_accuracy = update, actual["accuracy"]
    selection = json.loads((directory / "best_selection.json").read_text(encoding="utf-8"))
    if (selection.get("update") != best_update or
            selection.get("validation_accuracy") != best_accuracy or
            selection.get("audit_checkpoint") != checkpoints[best_update]["file"] or
            manifest.get("selected_update") != best_update):
        raise ValueError(f"Best checkpoint selection disagrees with validation: {directory}")
    for adapter in ("best_adapter", "final_adapter"):
        if not (directory / adapter / "adapter_config.json").is_file() or not (
            directory / adapter / "adapter_model.safetensors"
        ).is_file():
            raise ValueError(f"Missing adapter {adapter}: {directory}")
    outcomes, test_metrics = _evaluation(directory / "test.jsonl.gz",
                                         plan["split"]["test"], max_steps=config.max_steps)
    record = json.loads((directory / "test_metrics.json").read_text(encoding="utf-8"))
    _check_metrics(record, test_metrics, directory)
    if (record.get("selected_update") != best_update or
            record.get("selected_validation_accuracy") != best_accuracy or
            record.get("selected_checkpoint") != checkpoints[best_update]["file"]):
        raise ValueError(f"Test did not use the selected checkpoint: {directory}")
    counts["test_generation_decisions"] = test_metrics["model_calls"]
    counts["test_generated_tokens"] = test_metrics["generated_tokens"]
    counts["test_invalid_problems"] = test_metrics["invalid_trajectories"]
    counts["test_step_limit_problems"] = test_metrics["step_limit_trajectories"]
    counts["test_problems"] = test_metrics["problems"]
    return manifest, outcomes, dict(counts)


def summarize(run_directories: list[Path], *, bootstrap_seed: int = 20260926,
              bootstrap_replicates: int = 10000) -> dict:
    runs: dict[tuple[str, int], dict[int, int]] = {}
    diagnostics: dict[tuple[str, int], dict] = {}
    plan_hashes = set()
    expected_pairs = None
    versions = None
    shared_plan = None
    for directory in run_directories:
        manifest, outcomes, counts = _load_run(directory)
        plan = manifest["plan"]
        planned = {(arm, seed) for arm in plan["config"]["arms"]
                   for seed in plan["config"]["training_seeds"]}
        if expected_pairs is None:
            expected_pairs = planned
            versions = manifest["versions"]
            shared_plan = plan
        if planned != expected_pairs or manifest["versions"] != versions:
            raise ValueError("Compared runs have different plans or library versions")
        key = (manifest["arm"], manifest["seed"])
        if key in runs:
            raise ValueError(f"Duplicate arm and seed: {key}")
        runs[key] = outcomes
        diagnostics[key] = counts
        plan_hashes.add(manifest["plan_sha256"])
    if len(plan_hashes) != 1 or set(runs) != expected_pairs:
        raise ValueError("The complete paired experiment is required for a summary")
    assert shared_plan is not None
    seeds = sorted(seed for arm, seed in runs if arm == "baseline")
    test_indices = shared_plan["split"]["test"]
    per_seed = [
        {"seed": seed,
         "baseline": sum(runs[("baseline", seed)].values()) / len(test_indices),
         "two_step": sum(runs[("two_step", seed)].values()) / len(test_indices)}
        for seed in seeds
    ]
    for row in per_seed:
        row["difference"] = row["two_step"] - row["baseline"]
    differences = [
        sum(runs[("two_step", seed)][index] - runs[("baseline", seed)][index]
            for seed in seeds) / len(seeds)
        for index in test_indices
    ]
    rng = Random(bootstrap_seed)
    samples = sorted(
        sum(differences[rng.randrange(len(differences))]
            for _ in differences) / len(differences)
        for _ in range(bootstrap_replicates)
    )
    return {
        "schema_version": 1,
        "complete_planned_experiment": True,
        "run_count": len(runs),
        "plan_sha256": next(iter(plan_hashes)),
        "test_problems": test_indices,
        "test_problems_shortest_over_16": shared_plan["split"]["over_16_steps"]["test"],
        "seeds": seeds,
        "per_seed": per_seed,
        "mean_baseline": sum(row["baseline"] for row in per_seed) / len(seeds),
        "mean_two_step": sum(row["two_step"] for row in per_seed) / len(seeds),
        "paired_difference": sum(differences) / len(differences),
        "paired_problem_bootstrap_95_percent_interval": [
            samples[int(0.025*bootstrap_replicates)],
            samples[int(0.975*bootstrap_replicates)-1],
        ],
        "per_run_budget_and_gates": [
            {"arm": arm, "seed": seed, **diagnostics[(arm, seed)]}
            for arm in ("baseline", "two_step") for seed in seeds
        ],
        "budget_unit": "One generation decision is one model response for one Blocksworld state; scoring forwards excluded.",
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "interpretation": "Problem bootstrap holds the recorded training seeds fixed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("run_directories", nargs="+", type=Path)
    args = parser.parse_args()
    result = summarize(args.run_directories)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
