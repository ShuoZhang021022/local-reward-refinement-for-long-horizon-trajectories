"""Paired on-policy GRPO training for PlanBench six-block Blocksworld.

The advantage, gate, clipping, KL, and reduction code is shared with the
already confirmed Game24 implementation. Task generation and interaction are
Blocksworld-specific and locked in this package.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import traceback
from typing import Any

from game24_experiment.checkpoints import save_training_state
from game24_experiment.io import append_jsonl, write_json, write_jsonl_gzip
from game24_experiment.method import compute_advantages
from game24_experiment.train import (
    _attach_old_and_reference_logprobs, _diagnostic_metrics, _update, _visits,
)

from .config import ExperimentConfig
from .data import load_instances, split_instances


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_state() -> dict:
    def call(*args: str) -> str | None:
        result = subprocess.run(["git", *args], capture_output=True, text=True,
                                check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": call("rev-parse", "HEAD"), "status": call("status", "--short")}


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "blocksworld_experiment").glob("*.py"))
    files += [root / "blocksworld_experiment" / "experiment.json",
              root / "blocksworld_experiment" / "README.md",
              root / "blocksworld_experiment" / "PROTOCOL.md",
              root / "README.md", root / "requirements.txt"]
    files += sorted((root / "game24_experiment").glob("*.py"))
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def _config_from_plan(plan: dict) -> ExperimentConfig:
    if plan.get("schema_version") != 1:
        raise ValueError("Recreate the Blocksworld experiment plan")
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
        raise ValueError("Plan must pin checkpoint EOS and padding IDs")
    return config


def prepare(config_path: Path, output: Path) -> None:
    from huggingface_hub import HfApi
    from transformers import GenerationConfig
    from game24_experiment.modeling import checkpoint_special_tokens

    config = ExperimentConfig.from_json(config_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Plan directory is not empty: {output}")
    dataset_dir = Path(config.dataset_dir)
    instances, dataset_manifest = load_instances(dataset_dir,
                                                 expected_count=config.dataset_size)
    split = split_instances(
        instances, seed=config.split_seed, train_size=config.train_size,
        validation_size=config.validation_size, test_size=config.test_size,
    )
    revision = (HfApi().model_info(config.model_id).sha
                if config.model_revision == "resolve_at_prepare_and_lock_sha"
                else config.model_revision)
    if revision is None or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Could not resolve an immutable model revision")
    generation_config = GenerationConfig.from_pretrained(config.model_id,
                                                          revision=revision)
    effective = config.record()
    effective["model_revision"] = revision
    initial_branches = Counter(len(x.problem.initial.legal_actions()) for x in instances)
    plan = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "config": effective,
        "tokenizer_revision": revision,
        "special_tokens": checkpoint_special_tokens(generation_config),
        "dataset_manifest_sha256": hashlib.sha256(
            (dataset_dir / "manifest.json").read_bytes()).hexdigest(),
        "dataset_source": {key: dataset_manifest[key] for key in
                           ("source_repository", "source_revision", "generator_command",
                            "generator_files", "candidate_calls", "rejected_trivial_initial_goal",
                            "rejected_duplicate_initial_goal")},
        "split": split,
        "initial_legal_action_counts": dict(sorted(initial_branches.items())),
        "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "git": _git_state(),
        "source_hashes": _source_hashes(),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "plan.json", plan)
    write_json(output / "split.json", split)
    print(f"Prepared {output / 'plan.json'}; train={len(split['train'])}, "
          f"validation={len(split['validation'])}, test={len(split['test'])}; "
          f"shortest>16: {sum(split['over_16_steps'].values())}")


def _stats_record(trajectories: list[Any], batch: Any, *, arm: str) -> dict:
    gates_by_state: dict = defaultdict(list)
    for gate in batch.gates.values():
        gates_by_state[gate.parent_state].append(gate)
    children: dict = {}
    available: dict = {}
    for trajectory in trajectories:
        for decision in trajectory.decisions:
            transition = decision.transition
            state = transition.before
            children.setdefault(state, {})[transition.action_key] = transition.after
            earlier = available.setdefault(state, transition.legal_action_count)
            if earlier != transition.legal_action_count:
                raise ValueError("Same full state has inconsistent legal-action counts")
    state_rows = []
    for state, info in batch.states.items():
        record = info.record()
        record["action_children"] = children.get(state, {})
        record["distinct_legal_actions_available"] = available[state]
        gates = gates_by_state[state]
        positions = [batch.advantages[visit_id] for visit_id in info.visits]
        passed = sum(gate.passed for gate in gates)
        lambda_eligible = sum(row.B for row in positions)
        beta_eligible = sum(row.P and row.selected_second for row in positions)
        overlap_eligible = sum(row.overlap for row in positions)
        record["gate_summary"] = {
            "visit_count": len(positions),
            "parent_zero_variance": info.sigma == 0,
            "first_selected_visit_count": sum(row.selected_first for row in positions),
            "candidate_action_count": len(gates),
            "passes_five_action_count": sum(gate.passes_five for gate in gates),
            "passes_delta_action_count": sum(gate.passes_delta for gate in gates),
            "passed_action_count": passed,
            "passed_candidate_action_fraction": passed / len(gates) if gates else None,
            "gate_reason_counts": dict(Counter(gate.reason for gate in gates)),
            "lambda_eligible_visit_count": lambda_eligible,
            "beta_eligible_visit_count": beta_eligible,
            "overlap_eligible_visit_count": overlap_eligible,
            "lambda_applied_visit_count": lambda_eligible if arm == "two_step" else 0,
            "beta_applied_visit_count": beta_eligible if arm == "two_step" else 0,
            "overlap_applied_visit_count": overlap_eligible if arm == "two_step" else 0,
            "nonzero_applied_advantage_visit_count": sum(
                (row.final if arm == "two_step" else row.base) != 0 for row in positions),
            "single_visit_action_count": sum(len(ids) == 1
                                             for ids in info.action_visits.values()),
        }
        successors: dict = defaultdict(list)
        for action, child in children.get(state, {}).items():
            if child is not None:
                successors[child].append(action)
        record["same_successor_action_groups"] = [
            {"child_state": child, "actions": actions,
             "action_visit_counts": {action: len(info.action_visits[action])
                                     for action in actions}}
            for child, actions in successors.items() if len(actions) > 1
        ]
        state_rows.append(record)
    return {
        "schema_version": 1,
        "arm": arm,
        "gate_usage": "applied" if arm == "two_step" else "counterfactual_diagnostic_only",
        "states": state_rows,
        "first_selected_visit_ids": sorted(batch.selected_first_visits),
        "gates": [gate.record() for gate in batch.gates.values()],
        "advantages": [
            {**row.record(), "applied_advantage": row.final if arm == "two_step" else row.base}
            for row in batch.advantages.values()
        ],
    }


def _evaluate(runner: Any, instances: list[Any], indices: list[int], *,
              label: str, output_path: Path) -> dict:
    selected = [(index, instances[index]) for index in indices]
    trajectories = runner.rollout(selected, repetitions=1, label=label, do_sample=False)
    write_jsonl_gzip(output_path, (trajectory.record() for trajectory in trajectories))
    success = sum(trajectory.reward for trajectory in trajectories)
    invalid = sum(str(trajectory.game.termination).startswith("invalid:")
                  for trajectory in trajectories)
    return {
        "successes": success,
        "problems": len(trajectories),
        "accuracy": success / len(trajectories),
        "invalid_trajectories": invalid,
        "step_limit_trajectories": sum(trajectory.game.termination == "step_limit"
                                       for trajectory in trajectories),
        "model_calls": sum(len(trajectory.decisions) for trajectory in trajectories),
        "generated_tokens": sum(len(decision.completion_ids)
                                for trajectory in trajectories
                                for decision in trajectory.decisions),
    }


def run(plan_path: Path, output: Path, *, arm: str, seed: int) -> None:
    import importlib.metadata
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
        raise ValueError("Source files changed after prepare")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {output}")
    dataset_dir = Path(config.dataset_dir)
    dataset_manifest_sha256 = hashlib.sha256(
        (dataset_dir / "manifest.json").read_bytes()).hexdigest()
    if dataset_manifest_sha256 != plan["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed after prepare")
    instances, _ = load_instances(dataset_dir, expected_count=config.dataset_size)
    split = split_instances(
        instances, seed=config.split_seed, train_size=config.train_size,
        validation_size=config.validation_size, test_size=config.test_size,
    )
    if split != plan["split"]:
        raise ValueError("Dataset split changed after prepare")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal training requires a CUDA GPU; no run was started")

    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_bytes(plan_bytes)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    manifest = {
        "schema_version": 1,
        "started_utc": _utc_now(),
        "arm": arm,
        "seed": seed,
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "plan": plan,
        "git": _git_state(),
        "source_hashes": _source_hashes(),
        "versions": {package: importlib.metadata.version(package) for package in
                     ("torch", "transformers", "peft", "huggingface_hub")},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic},
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory,
        "tokenizer_revision": plan["tokenizer_revision"],
        "audit_checkpoint_policy": "initial_and_every_update",
        "status": "running",
    }
    write_json(output / "manifest.json", manifest)

    tokenizer = AutoTokenizer.from_pretrained(config.model_id,
                                               revision=plan["tokenizer_revision"])
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Selected tokenizer has no EOS token")
    base = AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=config.model_revision, dtype=torch.bfloat16,
    ).to(device)
    lora = LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules), bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    model.config.use_cache = True
    model.enable_input_require_grads()
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters")
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable], lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    runner = PolicyRunner(
        model, tokenizer, max_new_tokens=config.max_new_tokens,
        generation_batch_size=config.generation_batch_size, device=device,
        max_steps=config.max_steps,
    )
    if runner.special_tokens != plan["special_tokens"]:
        raise ValueError("Checkpoint EOS/padding IDs differ from the locked plan")
    manifest["trainable_parameters"] = {
        "count": sum(parameter.numel() for _, parameter in trainable),
        "names": [name for name, _ in trainable],
    }
    manifest["generation_config_from_checkpoint"] = model.generation_config.to_dict()
    manifest["generation_overrides"] = {
        "training": runner.generation_options(do_sample=True),
        "evaluation": runner.generation_options(do_sample=False),
    }
    manifest["optimizer_effective_parameter_groups"] = [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]
    manifest["lora_effective_config"] = {
        key: sorted(value) if isinstance(value, set) else value
        for key, value in lora.to_dict().items()
    }
    write_json(output / "manifest.json", manifest)

    best_accuracy = -1.0
    best_update = -1
    best_weights: dict[str, torch.Tensor] = {}

    def validation(update: int) -> None:
        nonlocal best_accuracy, best_update, best_weights
        metrics = _evaluate(
            runner, instances, plan["split"]["validation"], label=f"val:u{update}",
            output_path=output / f"validation_{update:04d}.jsonl.gz",
        )
        metrics.update({"update": update, "utc": _utc_now()})
        append_jsonl(output / "validations.jsonl", metrics)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_update = update
            best_weights = {name: parameter.detach().cpu().clone()
                            for name, parameter in trainable}
            model.save_pretrained(output / "best_adapter")
            write_json(output / "best_selection.json", {
                "update": update, "validation_accuracy": best_accuracy,
                "audit_checkpoint": f"checkpoints/state_{update:04d}.pt",
                "rule": "highest validation greedy success; earliest update on tie",
            })

    validation(0)
    ordered = list(plan["split"]["train"])
    random.Random(seed).shuffle(ordered)
    write_json(output / "train_order.json", {"indices": ordered, "seed": seed})

    def checkpoint(update: int) -> dict:
        record = save_training_state(
            output / "checkpoints" / f"state_{update:04d}.pt", model, optimizer,
            metadata={"update": update, "arm": arm, "seed": seed,
                      "plan_sha256": manifest["plan_sha256"],
                      "best_update": best_update,
                      "best_validation_accuracy": best_accuracy},
        )
        record["file"] = f"checkpoints/{record['file']}"
        append_jsonl(output / "checkpoints.jsonl", record)
        return record

    initial_checkpoint = checkpoint(0)
    updates_per_epoch = ((len(ordered) + config.problems_per_update - 1)
                         // config.problems_per_update)
    expected_updates = config.epochs * updates_per_epoch
    update_index = 0
    for epoch in range(config.epochs):
        if epoch:
            random.Random(seed + epoch).shuffle(ordered)
        for start in range(0, len(ordered), config.problems_per_update):
            update_index += 1
            indices = ordered[start:start + config.problems_per_update]
            selected = [(index, instances[index]) for index in indices]
            trajectories = runner.rollout(
                selected, repetitions=config.trajectories_per_problem,
                label=f"train:{arm}:{seed}:u{update_index}", do_sample=True,
            )
            _attach_old_and_reference_logprobs(runner, trajectories,
                                                config.scoring_batch_size)
            batch = compute_advantages(
                _visits(trajectories), omega=config.omega,
                lambda_bonus=config.lambda_bonus, beta=config.beta,
                seed=seed + update_index,
            )
            write_jsonl_gzip(
                output / f"rollouts_{update_index:04d}.jsonl.gz",
                (trajectory.record() for trajectory in trajectories),
            )
            state_stats = _stats_record(trajectories, batch, arm=arm)
            write_json(output / f"stats_{update_index:04d}.json", state_stats)
            metrics = _update(
                model, optimizer, trajectories, batch, arm=arm, config=config,
                pad_id=runner.pad_id, device=device,
                microbatch_size=config.gradient_microbatch_size,
                token_log_path=output / f"tokens_{update_index:04d}.jsonl.gz",
            )
            metrics.update(_diagnostic_metrics(trajectories, batch, arm=arm))
            metrics["states_with_five_available_legal_actions"] = sum(
                state["distinct_legal_actions_available"] >= 5
                for state in state_stats["states"]
            )
            metrics.update({
                "update": update_index, "epoch": epoch,
                "problem_indices": indices, "utc": _utc_now(),
                "old_policy_checkpoint": f"checkpoints/state_{update_index-1:04d}.pt",
                "new_policy_checkpoint": f"checkpoints/state_{update_index:04d}.pt",
            })
            if (update_index % config.validation_every_updates == 0 or
                    start + config.problems_per_update >= len(ordered)):
                validation(update_index)
            checkpoint_record = checkpoint(update_index)
            metrics["new_checkpoint_sha256"] = checkpoint_record["sha256"]
            append_jsonl(output / "updates.jsonl", metrics)
            print(f"update={update_index}/{expected_updates} "
                  f"success={metrics['successful_trajectories']}/{len(trajectories)} "
                  f"gates={metrics['passed_gate_count']}/{metrics['candidate_gate_count']} "
                  f"lambda={metrics['lambda_visit_count']} "
                  f"beta={metrics['beta_selected_visit_count']} "
                  f"flags={metrics['diagnostic_flags']}", flush=True)
            if update_index == 1:
                manifest["estimated_audit_checkpoint_bytes"] = (
                    initial_checkpoint["bytes"] +
                    expected_updates * checkpoint_record["bytes"]
                )
                write_json(output / "manifest.json", manifest)

    model.save_pretrained(output / "final_adapter")
    with torch.no_grad():
        for name, parameter in trainable:
            parameter.copy_(best_weights[name].to(device))
    test_metrics = _evaluate(
        runner, instances, plan["split"]["test"], label="test:selected",
        output_path=output / "test.jsonl.gz",
    )
    write_json(output / "test_metrics.json", {
        **test_metrics,
        "selected_update": best_update,
        "selected_validation_accuracy": best_accuracy,
        "selected_checkpoint": f"checkpoints/state_{best_update:04d}.pt",
    })
    manifest["status"] = "complete"
    manifest["finished_utc"] = _utc_now()
    manifest["selected_update"] = best_update
    manifest["completed_updates"] = update_index
    write_json(output / "manifest.json", manifest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    job = sub.add_parser("run")
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
                manifest["status"] = "failed"
                manifest["failed_utc"] = _utc_now()
                manifest["failure_traceback"] = traceback.format_exc()
                write_json(manifest_path, manifest)
            raise


if __name__ == "__main__":
    main()
