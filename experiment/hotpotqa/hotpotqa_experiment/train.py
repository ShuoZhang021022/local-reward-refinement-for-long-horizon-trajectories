"""Paired on-policy training for the five-read HotpotQA distractor variant."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import traceback
from typing import Any

from game24_experiment.checkpoints import file_sha256, save_training_state
from game24_experiment.io import append_jsonl, write_json, write_jsonl_gzip
from game24_experiment.method import compute_advantages
from game24_experiment.train import _attach_old_and_reference_logprobs, _update, _visits

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


def _evaluate(runner: Any, questions: dict[str, Any], ids: list[str], *,
              label: str, output_path: Path) -> dict:
    if not ids:
        raise ValueError("Evaluation split cannot be empty")
    all_trajectories = []
    for start in range(0, len(ids), runner.generation_batch_size):
        chunk_ids = ids[start:start + runner.generation_batch_size]
        indexed = [(start + index, questions[question_id])
                   for index, question_id in enumerate(chunk_ids)]
        all_trajectories.extend(runner.rollout(indexed, repetitions=1,
                                               label=label, do_sample=False))
    write_jsonl_gzip(output_path, (item.record() for item in all_trajectories))
    known_coverages = [item.support_document_coverage for item in all_trajectories
                       if item.support_document_coverage is not None]
    return {
        "question_count": len(all_trajectories),
        "answer_em": sum(item.exact_match for item in all_trajectories)
                     / len(all_trajectories),
        "answer_f1": float(sum((item.reward for item in all_trajectories), 0)
                           / len(all_trajectories)),
        "mean_read_count": sum(len(item.session.read_order)
                               for item in all_trajectories) / len(all_trajectories),
        "gold_support_document_coverage_mean": (
            float(sum(known_coverages, 0) / len(known_coverages))
            if known_coverages else None),
        "full_gold_support_document_coverage_fraction": (
            sum(value == 1 for value in known_coverages) / len(known_coverages)
            if known_coverages else None),
        "invalid_trajectories": sum(item.session.failed for item in all_trajectories),
        "model_calls": sum(item.record()["model_calls"] for item in all_trajectories),
        "generated_tokens": sum(item.record()["generated_tokens"]
                                for item in all_trajectories),
    }


def run(plan_path: Path, output: Path, *, arm: str, seed: int) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .modeling import PolicyRunner

    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    config = _config_from_plan(plan)
    if arm not in config.arms or seed not in config.training_seeds:
        raise ValueError("Arm or seed is not in the locked plan")
    if plan["source_hashes"] != _source_hashes():
        raise ValueError("Source files changed after the plan was locked")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {output}")
    train, dev, data_hashes = load_train_and_dev(Path(config.train_json),
                                                 Path(config.dev_json))
    validate_official_counts(train, dev)
    if data_hashes != plan["data_hashes"]:
        raise ValueError("Dataset files changed after prepare")
    split = split_train_validation(train, dev, seed=config.split_seed,
                                   validation_fraction=config.validation_fraction)
    if split != plan["split"]:
        raise ValueError("Dataset split changed after prepare")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal training requires a CUDA GPU; no run started")
    questions = {item.question_id: item for item in train + dev}
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_bytes(plan_bytes)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    manifest = {
        "schema_version": 1, "status": "running", "started_utc": _utc_now(),
        "arm": arm, "seed": seed,
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "plan": plan, "git": _git_state(), "source_hashes": _source_hashes(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "peft", "huggingface_hub")},
        "runtime": {"python": platform.python_version(),
                    "platform": platform.platform(), "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory,
        "audit_checkpoint_policy": "initial_and_every_update",
    }
    write_json(output / "manifest.json", manifest)

    tokenizer = AutoTokenizer.from_pretrained(config.model_id,
                                               revision=plan["tokenizer_revision"])
    base = AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=config.model_revision, dtype=torch.bfloat16,
    ).to(device)
    lora = LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha,
                      lora_dropout=config.lora_dropout,
                      target_modules=list(config.lora_target_modules), bias="none",
                      task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)
    model.config.use_cache = True
    model.enable_input_require_grads()
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters")
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable],
                                  lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    runner = PolicyRunner(model, tokenizer,
                          max_new_tokens=config.selection_max_new_tokens,
                          generation_batch_size=config.generation_batch_size,
                          device=device)
    if config.answer_max_new_tokens != runner.max_new_tokens:
        raise ValueError("Answer generation limit is not implemented separately")
    if runner.special_tokens != plan["special_tokens"]:
        raise ValueError("Checkpoint special tokens differ from the locked plan")
    manifest["trainable_parameters"] = {
        "count": sum(parameter.numel() for _, parameter in trainable),
        "names": [name for name, _ in trainable]}
    manifest["generation_config_from_checkpoint"] = model.generation_config.to_dict()
    manifest["generation_overrides"] = {
        "training_selection_and_answer": runner.generation_options(do_sample=True),
        "evaluation_selection_and_answer": runner.generation_options(do_sample=False),
    }
    manifest["optimizer_effective_parameter_groups"] = [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups]
    write_json(output / "manifest.json", manifest)

    best_f1 = -1.0
    best_update = -1
    best_weights: dict[str, torch.Tensor] = {}

    def validation(update: int) -> None:
        nonlocal best_f1, best_update, best_weights
        metrics = _evaluate(runner, questions, split["validation_ids"],
                            label=f"val:u{update}",
                            output_path=output / f"validation_{update:05d}.jsonl.gz")
        append_jsonl(output / "validations.jsonl",
                     {**metrics, "update": update, "utc": _utc_now()})
        if metrics["answer_f1"] > best_f1:
            best_f1 = metrics["answer_f1"]
            best_update = update
            best_weights = {name: parameter.detach().cpu().clone()
                            for name, parameter in trainable}
            model.save_pretrained(output / "best_adapter")
            write_json(output / "best_selection.json", {
                "update": update, "validation_answer_f1": best_f1,
                "rule": config.checkpoint_selection,
                "audit_checkpoint": f"checkpoints/state_{update:05d}.pt",
            })

    validation(0)

    def checkpoint(update: int) -> dict:
        result = save_training_state(
            output / "checkpoints" / f"state_{update:05d}.pt", model, optimizer,
            metadata={"update": update, "arm": arm, "seed": seed,
                      "plan_sha256": manifest["plan_sha256"],
                      "best_update": best_update, "best_validation_answer_f1": best_f1},
        )
        result["file"] = f"checkpoints/{result['file']}"
        append_jsonl(output / "checkpoints.jsonl", result)
        return result

    checkpoint(0)
    ordered = list(split["train_ids"])
    random.Random(seed).shuffle(ordered)
    write_json(output / "train_order.json", {"question_ids": ordered, "seed": seed})
    update_index = 0
    expected_updates = config.epochs * (
        (len(ordered) + config.questions_per_update - 1) // config.questions_per_update)
    for epoch in range(config.epochs):
        if epoch:
            random.Random(seed + epoch).shuffle(ordered)
        for start in range(0, len(ordered), config.questions_per_update):
            update_index += 1
            question_ids = ordered[start:start + config.questions_per_update]
            selected = [(index, questions[question_id])
                        for index, question_id in enumerate(question_ids)]
            trajectories = runner.rollout(
                selected, repetitions=config.trajectories_per_question,
                label=f"train:{arm}:{seed}:u{update_index}", do_sample=True)
            _attach_old_and_reference_logprobs(runner, trajectories,
                                               config.scoring_batch_size)
            batch = compute_advantages(
                _visits(trajectories), omega=config.omega,
                lambda_bonus=config.lambda_bonus, beta=config.beta,
                seed=seed + update_index)
            rollout_path = output / f"rollouts_{update_index:05d}.jsonl.gz"
            stats_path = output / f"stats_{update_index:05d}.json"
            token_path = output / f"tokens_{update_index:05d}.jsonl.gz"
            write_jsonl_gzip(rollout_path, (
                _trajectory_record_with_gate(item, batch, arm=arm)
                for item in trajectories))
            write_json(stats_path, _stats_record(trajectories, batch, arm=arm))
            metrics = _update(model, optimizer, trajectories, batch,
                              arm=arm, config=config, pad_id=runner.pad_id,
                              device=device,
                              microbatch_size=config.gradient_microbatch_size,
                              token_log_path=token_path)
            metrics.update(_diagnostic_metrics(trajectories, batch, arm=arm))
            metrics["update_artifacts"] = {
                artifact.name: {"bytes": artifact.stat().st_size,
                                "sha256": file_sha256(artifact)}
                for artifact in (rollout_path, stats_path, token_path)}
            metrics.update({"update": update_index, "epoch": epoch,
                            "question_ids": question_ids, "utc": _utc_now(),
                            "old_policy_checkpoint":
                                f"checkpoints/state_{update_index - 1:05d}.pt",
                            "new_policy_checkpoint":
                                f"checkpoints/state_{update_index:05d}.pt"})
            if (update_index % config.validation_every_updates == 0 or
                    start + config.questions_per_update >= len(ordered)):
                validation(update_index)
            checkpoint_record = checkpoint(update_index)
            metrics["new_checkpoint_sha256"] = checkpoint_record["sha256"]
            append_jsonl(output / "updates.jsonl", metrics)
            print(f"update={update_index}/{expected_updates} "
                  f"answer_f1={metrics['answer_f1_mean']:.4f} "
                  f"gates={metrics['passed_gate_count']}/{metrics['candidate_gate_count']}",
                  flush=True)
    model.save_pretrained(output / "final_adapter")
    with torch.no_grad():
        for name, parameter in trainable:
            parameter.copy_(best_weights[name].to(device))
    test_metrics = _evaluate(runner, questions, split["test_ids"],
                             label="test:selected",
                             output_path=output / "test.jsonl.gz")
    write_json(output / "test_metrics.json", {
        **test_metrics, "selected_update": best_update,
        "selected_validation_answer_f1": best_f1,
        "selected_checkpoint": f"checkpoints/state_{best_update:05d}.pt",
    })
    manifest.update({"status": "complete", "finished_utc": _utc_now(),
                     "selected_update": best_update,
                     "completed_updates": update_index})
    write_json(output / "manifest.json", manifest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Lock dataset, model, config, and source")
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    job = sub.add_parser("run", help="Run one arm and seed on CUDA")
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
            if not manifest_existed and manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.update({"status": "failed", "failed_utc": _utc_now(),
                                 "failure_traceback": traceback.format_exc()})
                write_json(manifest_path, manifest)
            raise


if __name__ == "__main__":
    main()
