"""Compare completed paired Game24 runs by puzzle and training seed."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from random import Random

from .checkpoints import file_sha256
from .data import load_puzzles
from .env import Game24
from .io import write_json
from .train import _config_from_plan


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"Missing run record: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _count(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid nonnegative count for {label}: {value!r}")
    return value


def _read_evaluation(path: Path, expected_indices: list[int],
                     expected_numbers: dict[int, tuple[int, int, int, int]],
                     *, eos_ids: set[int], max_new_tokens: int) -> tuple[dict[int, int], dict]:
    if not path.is_file():
        raise ValueError(f"Missing evaluation trace: {path}")
    outcomes: dict[int, int] = {}
    calls = tokens = invalid = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            index = _count(row["puzzle_index"], "puzzle_index")
            if index in outcomes:
                raise ValueError(f"Duplicate evaluation puzzle {index}: {path}")
            reward = row["reward"]
            if type(reward) is not int or reward not in (0, 1):
                raise ValueError(f"Invalid evaluation reward at puzzle {index}: {path}")
            decisions = row["decisions"]
            model_calls = _count(row["model_calls"], "model_calls")
            generated_tokens = _count(row["generated_tokens"], "generated_tokens")
            if (model_calls != len(decisions) or model_calls not in (1, 2, 3)
                    or row["steps"] != model_calls
                    or generated_tokens != sum(len(step["completion_ids"]) for step in decisions)
                    or generated_tokens < model_calls):
                raise ValueError(f"Inconsistent evaluation trajectory at puzzle {index}: {path}")
            if index not in expected_numbers or row.get("numbers") != list(expected_numbers[index]):
                raise ValueError(f"Evaluation puzzle numbers disagree with source at {index}: {path}")
            game = Game24(expected_numbers[index])
            for decision in decisions:
                ids = decision["completion_ids"]
                ended = decision["ended_by_eos"]
                if (not isinstance(ids, list) or not ids or len(ids) > max_new_tokens
                        or any(type(token) is not int for token in ids)
                        or type(ended) is not bool
                        or decision["generated_token_mask"] != [1] * len(ids)
                        or (ended and (ids[-1] not in eos_ids
                                       or any(token in eos_ids for token in ids[:-1])))
                        or (not ended and (len(ids) != max_new_tokens
                                           or any(token in eos_ids for token in ids)))
                        or decision["termination_token_id"] != (ids[-1] if ended else None)
                        or game.terminal):
                    raise ValueError(f"Invalid evaluation completion at puzzle {index}: {path}")
                recorded = decision["transition"]
                raw = recorded["raw_output"]
                actual = game.step(raw) if ended else game.fail(raw, "generation_limit")
                if recorded != json.loads(json.dumps(asdict(actual))):
                    raise ValueError(f"Evaluation transition disagrees with Game24 at {index}: {path}")
            termination = row["termination"]
            if (reward != game.reward or termination != game.termination
                    or row["final_expression"] != decisions[-1]["transition"]["expression"]):
                raise ValueError(f"Evaluation result disagrees with Game24 at puzzle {index}: {path}")
            outcomes[index] = reward
            calls += model_calls
            tokens += generated_tokens
            invalid += str(termination).startswith("invalid:")
    if sorted(outcomes) != sorted(expected_indices):
        raise ValueError(f"Evaluation puzzle set mismatch: {path}")
    metrics = {"successes": sum(outcomes.values()), "puzzles": len(outcomes),
               "accuracy": sum(outcomes.values()) / len(outcomes),
               "invalid_trajectories": invalid, "model_calls": calls,
               "generated_tokens": tokens}
    return outcomes, metrics


def _check_metrics(record: dict, actual: dict, path: Path) -> None:
    if any(record.get(key) != value for key, value in actual.items()):
        raise ValueError(f"Evaluation metrics disagree with trace: {path}")


def _add_rates(counts: dict) -> dict:
    result = dict(counts)
    result["training_invalid_fraction"] = (
        result["training_invalid_trajectories"] / result["training_trajectories"]
    )
    result["test_invalid_fraction"] = (
        result["test_invalid_puzzles"] / result["test_puzzles"]
    )
    result["passed_candidate_action_fraction"] = (
        result["passed_gate_actions"] / result["candidate_gate_actions"]
        if result["candidate_gate_actions"] else None
    )
    result["lambda_applied_decision_fraction"] = (
        result["lambda_applied_visits"] / result["training_generation_decisions"]
    )
    result["beta_applied_decision_fraction"] = (
        result["beta_applied_visits"] / result["training_generation_decisions"]
    )
    return result


def _load_run(path: Path) -> tuple[dict, dict[int, int], dict]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError(f"Run predates the audited experiment schema: {path}")
    if manifest["status"] != "complete":
        raise ValueError(f"Incomplete run: {path}")
    plan_path = path / "plan.json"
    if not plan_path.is_file() or file_sha256(plan_path) != manifest["plan_sha256"]:
        raise ValueError(f"Missing or altered locked plan: {path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan != manifest["plan"]:
        raise ValueError(f"Manifest plan differs from locked plan: {path}")
    config = _config_from_plan(plan)
    dataset_path = Path(config.dataset_csv)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parents[1] / dataset_path
    puzzles, data_hash = load_puzzles(dataset_path)
    if data_hash != plan["split"]["source_sha256"]:
        raise ValueError(f"Evaluation source data disagrees with locked plan: {path}")
    expected_numbers = {index: puzzle.numbers for index, puzzle in enumerate(puzzles)}
    eos_ids = set(plan["special_tokens"]["eos_token_ids"])
    if manifest["arm"] not in config.arms or manifest["seed"] not in config.training_seeds:
        raise ValueError(f"Unexpected arm or seed: {path}")
    updates_per_epoch = (
        (len(plan["split"]["train"]) + config.puzzles_per_update - 1)
        // config.puzzles_per_update
    )
    expected_updates = config.epochs * updates_per_epoch
    if manifest.get("completed_updates") != expected_updates:
        raise ValueError(f"Run did not complete the planned updates: {path}")
    checkpoints = _read_jsonl(path / "checkpoints.jsonl")
    if [row.get("update") for row in checkpoints] != list(range(expected_updates + 1)):
        raise ValueError(f"Incomplete or unordered checkpoint records: {path}")
    for update, record in enumerate(checkpoints):
        relative = f"checkpoints/state_{update:04d}.pt"
        snapshot = path / relative
        if (record.get("file") != relative or record.get("arm") != manifest["arm"]
                or record.get("seed") != manifest["seed"]
                or record.get("plan_sha256") != manifest["plan_sha256"]
                or not snapshot.is_file() or snapshot.stat().st_size != record.get("bytes")
                or file_sha256(snapshot) != record.get("sha256")):
            raise ValueError(f"Missing or altered audit checkpoint {relative}: {path}")

    order = json.loads((path / "train_order.json").read_text(encoding="utf-8"))
    ordered = order["indices"]
    if (order.get("seed") != manifest["seed"]
            or len(ordered) != len(plan["split"]["train"])
            or sorted(ordered) != sorted(plan["split"]["train"])):
        raise ValueError(f"Training order disagrees with plan: {path}")
    updates = _read_jsonl(path / "updates.jsonl")
    if [row.get("update") for row in updates] != list(range(1, expected_updates + 1)):
        raise ValueError(f"Incomplete or unordered update records: {path}")
    counts = Counter()
    for epoch in range(config.epochs):
        if epoch:
            Random(manifest["seed"] + epoch).shuffle(ordered)
        for start in range(0, len(ordered), config.puzzles_per_update):
            update = epoch * updates_per_epoch + start // config.puzzles_per_update + 1
            row = updates[update - 1]
            indices = ordered[start:start + config.puzzles_per_update]
            if (row.get("epoch") != epoch or row.get("puzzle_indices") != indices
                    or row.get("old_policy_checkpoint") != checkpoints[update - 1]["file"]
                    or row.get("new_policy_checkpoint") != checkpoints[update]["file"]
                    or row.get("new_checkpoint_sha256") != checkpoints[update]["sha256"]):
                raise ValueError(f"Update {update} disagrees with plan or checkpoints: {path}")
            trajectories = _count(row["trajectory_count"], "trajectory_count")
            decisions = _count(row["decision_count"], "decision_count")
            if (trajectories != len(indices) * config.trajectories_per_puzzle
                    or not trajectories <= decisions <= 3 * trajectories):
                raise ValueError(f"Update {update} has an invalid generation budget: {path}")
            keys = {
                "training_trajectories": "trajectory_count",
                "training_generation_decisions": "decision_count",
                "training_generated_tokens": "generated_tokens",
                "training_invalid_trajectories": "invalid_trajectories",
                "candidate_gate_actions": "candidate_gate_count",
                "passed_gate_actions": "passed_gate_count",
                "lambda_eligible_visits": "lambda_eligible_visit_count",
                "beta_eligible_visits": "beta_eligible_visit_count",
                "lambda_applied_visits": "lambda_visit_count",
                "beta_applied_visits": "beta_selected_visit_count",
                "overlap_applied_visits": "overlap_visit_count",
            }
            for target, source in keys.items():
                counts[target] += _count(row[source], f"update {update} {source}")
            if (row["generated_tokens"] < decisions
                    or row["invalid_trajectories"] > trajectories
                    or row["passed_gate_count"] > row["candidate_gate_count"]):
                raise ValueError(f"Update {update} has inconsistent counts: {path}")
            expected_artifacts = (f"rollouts_{update:04d}.jsonl.gz",
                                  f"tokens_{update:04d}.jsonl.gz",
                                  f"stats_{update:04d}.json")
            artifacts = row.get("update_artifacts")
            if not isinstance(artifacts, dict) or set(artifacts) != set(expected_artifacts):
                raise ValueError(f"Update {update} lacks artifact checksums: {path}")
            for filename in expected_artifacts:
                artifact = path / filename
                recorded = artifacts[filename]
                if (not isinstance(recorded, dict) or not artifact.is_file()
                        or artifact.stat().st_size != recorded.get("bytes")
                        or file_sha256(artifact) != recorded.get("sha256")):
                    raise ValueError(f"Missing or altered update artifact {filename}: {path}")

    expected_validations = [0] + [
        update for update in range(1, expected_updates + 1)
        if update % config.validation_every_updates == 0 or update % updates_per_epoch == 0
    ]
    validations = _read_jsonl(path / "validations.jsonl")
    if [row.get("update") for row in validations] != expected_validations:
        raise ValueError(f"Missing or unordered validation records: {path}")
    best_update = None
    best_accuracy = -1.0
    for update, row in zip(expected_validations, validations):
        _, actual = _read_evaluation(
            path / f"validation_{update:04d}.jsonl.gz", plan["split"]["validation"],
            expected_numbers, eos_ids=eos_ids, max_new_tokens=config.max_new_tokens,
        )
        _check_metrics(row, actual, path)
        counts["validation_generation_decisions"] += actual["model_calls"]
        counts["validation_generated_tokens"] += actual["generated_tokens"]
        if actual["accuracy"] > best_accuracy:
            best_update, best_accuracy = update, actual["accuracy"]
    selection = json.loads((path / "best_selection.json").read_text(encoding="utf-8"))
    if (selection.get("update") != best_update
            or selection.get("validation_accuracy") != best_accuracy
            or selection.get("audit_checkpoint") != checkpoints[best_update]["file"]
            or manifest.get("selected_update") != best_update):
        raise ValueError(f"Best checkpoint does not follow validation rule: {path}")
    adapter_artifacts = manifest.get("adapter_artifacts")
    if not isinstance(adapter_artifacts, dict) or set(adapter_artifacts) != {
        "best_adapter", "final_adapter",
    }:
        raise ValueError(f"Missing adapter artifact checksums: {path}")
    for adapter in ("best_adapter", "final_adapter"):
        expected_files = adapter_artifacts[adapter]
        if not isinstance(expected_files, dict) or set(expected_files) != {
            "adapter_config.json", "adapter_model.safetensors",
        }:
            raise ValueError(f"Missing adapter artifact checksums for {adapter}: {path}")
        for filename, recorded in expected_files.items():
            artifact = path / adapter / filename
            if (not isinstance(recorded, dict) or not artifact.is_file()
                    or artifact.stat().st_size != recorded.get("bytes")
                    or file_sha256(artifact) != recorded.get("sha256")):
                raise ValueError(f"Missing or altered saved adapter {adapter}/{filename}: {path}")

    outcomes, actual_test = _read_evaluation(
        path / "test.jsonl.gz", plan["split"]["test"], expected_numbers,
        eos_ids=eos_ids, max_new_tokens=config.max_new_tokens,
    )
    if sorted(outcomes) != list(range(900, 1000)):
        raise ValueError(f"Test set mismatch: {path}")
    test_metrics = json.loads((path / "test_metrics.json").read_text(encoding="utf-8"))
    _check_metrics(test_metrics, actual_test, path)
    if (test_metrics.get("selected_update") != best_update
            or test_metrics.get("selected_validation_accuracy") != best_accuracy
            or test_metrics.get("selected_checkpoint") != checkpoints[best_update]["file"]):
        raise ValueError(f"Test did not use the selected checkpoint: {path}")
    counts["test_generation_decisions"] = actual_test["model_calls"]
    counts["test_generated_tokens"] = actual_test["generated_tokens"]
    counts["test_invalid_puzzles"] = actual_test["invalid_trajectories"]
    counts["test_puzzles"] = actual_test["puzzles"]
    return manifest, outcomes, _add_rates(counts)


def summarize(run_directories: list[Path], *, bootstrap_seed: int = 20260926,
              bootstrap_replicates: int = 10000) -> dict:
    runs: dict[tuple[str, int], dict[int, int]] = {}
    diagnostics: dict[tuple[str, int], dict] = {}
    plan_hashes: set[str] = set()
    expected_pairs: set[tuple[str, int]] | None = None
    versions: dict | None = None
    for path in run_directories:
        manifest, outcomes, run_diagnostics = _load_run(path)
        configuration = manifest["plan"]["config"]
        planned = {(arm, seed) for arm in configuration["arms"]
                   for seed in configuration["training_seeds"]}
        if expected_pairs is None:
            expected_pairs = planned
            versions = manifest["versions"]
        if planned != expected_pairs:
            raise ValueError("Compared runs specify different planned arms/seeds")
        if versions != manifest["versions"]:
            raise ValueError("Compared runs must use the same recorded library versions")
        key = (manifest["arm"], int(manifest["seed"]))
        if key in runs:
            raise ValueError(f"Duplicate arm/seed: {key}")
        runs[key] = outcomes
        diagnostics[key] = run_diagnostics
        plan_hashes.add(manifest["plan_sha256"])
    if len(plan_hashes) != 1:
        raise ValueError("Compared runs must use the same locked plan")
    if set(runs) != expected_pairs:
        raise ValueError(f"Incomplete planned experiment: missing={sorted((expected_pairs or set()) - set(runs))}")
    baseline_seeds = {seed for arm, seed in runs if arm == "baseline"}
    two_step_seeds = {seed for arm, seed in runs if arm == "two_step"}
    if not baseline_seeds or baseline_seeds != two_step_seeds:
        raise ValueError("Need paired baseline and two_step runs for each seed")
    seeds = sorted(baseline_seeds)
    per_seed = []
    for seed in seeds:
        base_rate = sum(runs[("baseline", seed)].values()) / 100
        two_rate = sum(runs[("two_step", seed)].values()) / 100
        per_seed.append({"seed": seed, "baseline": base_rate,
                         "two_step": two_rate, "difference": two_rate - base_rate})
    per_puzzle_difference = [
        sum(runs[("two_step", seed)][index] - runs[("baseline", seed)][index]
            for seed in seeds) / len(seeds)
        for index in range(900, 1000)
    ]
    mean_difference = sum(per_puzzle_difference) / 100
    by_arm_counts = {}
    for arm in ("baseline", "two_step"):
        by_arm_counts[arm] = _add_rates(Counter({
            key: sum(diagnostics[(arm, seed)][key] for seed in seeds)
            for key in diagnostics[(arm, seeds[0])]
            if type(diagnostics[(arm, seeds[0])][key]) is int
        }))
    rng = Random(bootstrap_seed)
    samples = sorted(
        sum(per_puzzle_difference[rng.randrange(100)] for _ in range(100)) / 100
        for _ in range(bootstrap_replicates)
    )
    return {
        "schema_version": 2,
        "complete_planned_experiment": True,
        "run_count": len(runs),
        "plan_sha256": next(iter(plan_hashes)),
        "seeds": seeds,
        "test_puzzles": list(range(900, 1000)),
        "per_seed": per_seed,
        "per_run_budget_and_gates": [
            {"arm": arm, "seed": seed, **diagnostics[(arm, seed)]}
            for arm in ("baseline", "two_step") for seed in seeds
        ],
        "by_arm_budget_and_gates": by_arm_counts,
        "budget_unit": "One generation decision is one model response for one Game24 state; scoring forwards are excluded.",
        "mean_baseline": sum(row["baseline"] for row in per_seed) / len(seeds),
        "mean_two_step": sum(row["two_step"] for row in per_seed) / len(seeds),
        "paired_difference": mean_difference,
        "paired_puzzle_bootstrap_95_percent_interval": [
            samples[int(0.025 * bootstrap_replicates)],
            samples[int(0.975 * bootstrap_replicates) - 1],
        ],
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "interpretation": "Bootstrap resamples puzzles while holding the recorded training seeds fixed.",
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
