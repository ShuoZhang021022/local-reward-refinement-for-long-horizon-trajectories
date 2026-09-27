"""Paired on-policy training for the five-read HotpotQA distractor variant."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
import os
from typing import Any

from game24_experiment.io import write_json

from .config import ExperimentConfig
from .data import load_train_and_dev, split_train_validation, validate_official_counts


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_state() -> dict:
    def call(*args: str) -> str | None:
        result = subprocess.run(["git", *args], capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": call("rev-parse", "HEAD"), "status": call("status", "--short")}


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "hotpotqa_experiment").glob("*.py"))
    files += [root / "hotpotqa_experiment" / "experiment.json",
              root / "hotpotqa_experiment" / "PROTOCOL.md",
              root / "two_step_gated_grpo_theory.md",
              root / "action_mean_grpo_baseline.md"]
    files += [root / "game24_experiment" / name for name in (
        "env.py", "method.py", "loss.py", "modeling.py", "train.py",
        "checkpoints.py", "io.py")]
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def _config_from_plan(plan: dict) -> ExperimentConfig:
    if plan.get("schema_version") != 1:
        raise ValueError("Recreate the HotpotQA plan with the current prepare command")
    raw = dict(plan["config"])
    for key in ("training_seeds", "arms", "lora_target_modules"):
        raw[key] = tuple(raw[key])
    config = ExperimentConfig(**raw)
    config.validate()
    revision = config.model_revision
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Plan must pin a 40-character model revision SHA")
    if plan.get("tokenizer_revision") != revision:
        raise ValueError("Tokenizer revision must match model revision")
    if not plan.get("special_tokens", {}).get("eos_token_ids"):
        raise ValueError("Plan must pin EOS and padding token IDs")
    return config


def prepare(config_path: Path, output: Path) -> None:
    from huggingface_hub import HfApi
    from transformers import GenerationConfig
    from game24_experiment.modeling import checkpoint_special_tokens

    config = ExperimentConfig.from_json(config_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Plan directory is not empty: {output}")
    train, dev, data_hashes = load_train_and_dev(Path(config.train_json),
                                                 Path(config.dev_json))
    validate_official_counts(train, dev)
    split = split_train_validation(train, dev, seed=config.split_seed,
                                   validation_fraction=config.validation_fraction)
    revision = (HfApi().model_info(config.model_id).sha
                if config.model_revision == "resolve_at_run_start_and_lock_sha"
                else config.model_revision)
    if revision is None or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Could not resolve immutable model revision")
    generation_config = GenerationConfig.from_pretrained(config.model_id,
                                                          revision=revision)
    effective = config.record()
    effective["model_revision"] = revision
    plan = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "config": effective,
        "tokenizer_revision": revision,
        "special_tokens": checkpoint_special_tokens(generation_config),
        "data_hashes": data_hashes,
        "split": split,
        "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "git": _git_state(),
        "source_hashes": _source_hashes(),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "plan.json", plan)
    write_json(output / "split.json", split)
    print(f"Prepared {output / 'plan.json'}; train={len(split['train_ids'])}, "
          f"validation={len(split['validation_ids'])}, dev test={len(split['test_ids'])}")


def _stats_record(trajectories: list[Any], batch: Any, *, arm: str) -> dict:
    available: dict[tuple[str, ...], int] = {}
    children: dict[tuple[str, ...], dict[str, tuple[str, ...] | None]] = {}
    for trajectory in trajectories:
        for decision in trajectory.decisions:
            transition = decision.transition
            state = transition.before
            count = len(transition.available_actions)
            if state in available and available[state] != count:
                raise ValueError("Same anchor has inconsistent available-action counts")
            available[state] = count
            by_action = children.setdefault(state, {})
            if (transition.action_key in by_action and
                    by_action[transition.action_key] != transition.after):
                raise ValueError("Same anchor and action have inconsistent child anchors")
            by_action[transition.action_key] = transition.after
    states = []
    for state, info in batch.states.items():
        record = info.record()
        record["selection_step"] = _selection_step(state)
        record["available_legal_actions"] = available[state]
        record["action_children"] = children.get(state, {})
        states.append(record)
    return {
        "schema_version": 1,
        "arm": arm,
        "gate_usage": "applied" if arm == "two_step" else "counterfactual_diagnostic_only",
        "states": states,
        "selected_first_visit_ids": sorted(batch.selected_first_visits),
        "gates": [{**gate.record(), "selection_step": _selection_step(gate.parent_state)}
                  for gate in batch.gates.values()],
        "advantages": [
            {**item.record(), "applied_advantage": item.final if arm == "two_step"
             else item.base}
            for item in batch.advantages.values()
        ],
    }


def _selection_step(anchor: tuple[str, ...]) -> int:
    """Choice after k reads is step k+1, irrespective of their order."""
    step = 1 + sum(item.startswith("read:") for item in anchor)
    if not 1 <= step <= 5:
        raise ValueError(f"Invalid HotpotQA selection anchor: {anchor}")
    return step


def _gate_by_step(batch: Any, *, arm: str) -> dict[str, dict]:
    """Count candidate-action and selected-visit gate rates separately."""
    result = {str(step): {
        "decision_visits": 0,
        "selected_first_visits": 0,
        "passing_selected_first_visits": 0,
        "candidate_gate_actions": 0,
        "nonterminal_candidate_gate_actions": 0,
        "passing_candidate_gate_actions": 0,
        "lambda_applied_visits": 0,
        "beta_applied_visits": 0,
    } for step in range(1, 6)}
    for state, info in batch.states.items():
        row = result[str(_selection_step(state))]
        row["decision_visits"] += len(info.visits)
        for visit_id in info.visits:
            advantage = batch.advantages[visit_id]
            row["selected_first_visits"] += int(advantage.selected_first)
            row["passing_selected_first_visits"] += int(advantage.B)
            if arm == "two_step":
                row["lambda_applied_visits"] += int(advantage.B)
                row["beta_applied_visits"] += int(
                    advantage.P and advantage.selected_second)
    for gate in batch.gates.values():
        row = result[str(_selection_step(gate.parent_state))]
        row["candidate_gate_actions"] += 1
        row["nonterminal_candidate_gate_actions"] += int(gate.child_state is not None)
        row["passing_candidate_gate_actions"] += int(gate.passed)
    for row in result.values():
        candidates = row["candidate_gate_actions"]
        nonterminal = row["nonterminal_candidate_gate_actions"]
        selected = row["selected_first_visits"]
        passed = row["passing_candidate_gate_actions"]
        row["candidate_gate_pass_rate"] = passed / candidates if candidates else None
        row["nonterminal_candidate_gate_pass_rate"] = (
            passed / nonterminal if nonterminal else None)
        row["selected_first_visit_gate_pass_rate"] = (
            row["passing_selected_first_visits"] / selected if selected else None)
    return result


def _trajectory_record_with_gate(trajectory: Any, batch: Any, *, arm: str) -> dict:
    """Join each training decision to its gate and applied advantage by visit ID."""
    record = trajectory.record()
    for decision in record["decisions"]:
        visit_id = decision["visit_id"]
        advantage = batch.advantages[visit_id]
        transition = decision["transition"]
        state = tuple(transition["before"])
        gate = batch.gates.get((state, transition["action_key"]))
        decision["selection_step"] = _selection_step(state)
        decision["advantage"] = advantage.record()
        decision["applied_advantage"] = (
            advantage.final if arm == "two_step" else advantage.base)
        decision["gate"] = (gate.record() if gate is not None else None)
    return record


def _diagnostic_metrics(trajectories: list[Any], batch: Any, *, arm: str) -> dict:
    reasons = Counter(gate.reason for gate in batch.gates.values())
    applied = [item.final if arm == "two_step" else item.base
               for item in batch.advantages.values()]
    reward_sum = sum((trajectory.reward for trajectory in trajectories), 0)
    coverages = [item.support_document_coverage for item in trajectories]
    known_coverages = [value for value in coverages if value is not None]
    return {
        "trajectory_count": len(trajectories),
        "answer_f1_mean": float(reward_sum / len(trajectories)),
        "answer_em_mean": sum(item.exact_match for item in trajectories) / len(trajectories),
        "read_count_mean": sum(len(item.session.read_order) for item in trajectories)
                           / len(trajectories),
        "gold_support_document_coverage_mean": (
            float(sum(known_coverages, 0) / len(known_coverages))
            if known_coverages else None),
        "full_gold_support_document_coverage_fraction": (
            sum(value == 1 for value in known_coverages) / len(known_coverages)
            if known_coverages else None),
        "invalid_trajectories": sum(item.session.failed for item in trajectories),
        "early_submissions": sum(item.session.answer_request_reason == "early_submit"
                                 for item in trajectories),
        "forced_answers": sum(item.session.answer_request_reason == "read_limit"
                              for item in trajectories),
        "model_calls": sum(len(item.decisions) + int(item.answer_generation is not None)
                           for item in trajectories),
        "generated_tokens": sum(item.record()["generated_tokens"]
                                for item in trajectories),
        "selection_tokens_in_objective": sum(len(decision.completion_ids)
                                             for item in trajectories
                                             for decision in item.decisions),
        "answer_tokens_excluded": sum(len(item.answer_generation.completion_ids)
                                      for item in trajectories
                                      if item.answer_generation is not None),
        "states": len(batch.states),
        "states_with_five_observed_actions": sum(
            info.observed_legal_actions >= 5 for info in batch.states.values()),
        "observed_legal_action_counts": [info.observed_legal_actions
                                         for info in batch.states.values()],
        "gate_reason_counts": dict(reasons),
        "passed_gate_count": reasons["passed"],
        "candidate_gate_count": len(batch.gates),
        "candidate_gate_pass_rate": (reasons["passed"] / len(batch.gates)
                                     if batch.gates else None),
        "gate_by_step": _gate_by_step(batch, arm=arm),
        "nonzero_applied_advantage_visits": sum(value != 0 for value in applied),
        "lambda_applied_visits": (sum(item.B for item in batch.advantages.values())
                                  if arm == "two_step" else 0),
        "beta_applied_visits": (sum(item.P and item.selected_second
                                   for item in batch.advantages.values())
                                if arm == "two_step" else 0),
    }


def run(plan_path: Path, output: Path, *, arm: str, seed: int) -> None:
    """Run one arm and seed synchronously on one eight-A100 node."""
    from .distributed import run as distributed_run

    distributed_run(plan_path, output, arm=arm, seed=seed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Lock dataset, model, config, and source")
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    job = sub.add_parser("run", help="Run one arm and seed on eight local A100 GPUs")
    job.add_argument("--plan", type=Path, required=True)
    job.add_argument("--output", type=Path, required=True)
    job.add_argument("--arm", choices=("baseline", "two_step"), required=True)
    job.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.config, args.output)
    else:
        manifest_path = args.output / "manifest.json"
        manifest_existed = manifest_path.exists()
        try:
            run(args.plan, args.output, arm=args.arm, seed=args.seed)
        except BaseException:
            if (not manifest_existed and manifest_path.is_file() and
                    os.environ.get("RANK", "0") == "0"):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.update({"status": "failed", "failed_utc": _utc_now(),
                                 "failure_traceback": traceback.format_exc()})
                write_json(manifest_path, manifest)
            raise


if __name__ == "__main__":
    main()
