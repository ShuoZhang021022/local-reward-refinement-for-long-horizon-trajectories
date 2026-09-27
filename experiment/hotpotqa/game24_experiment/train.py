"""Single-GPU on-policy Game24 training and evaluation.

This is intentionally a custom objective. Library GRPO defaults do not match
the two local method documents' action-mean, gate, overlap, and reduction rules.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
from typing import Any
from fractions import Fraction
import traceback

from .config import ExperimentConfig
from .data import dataset_provenance, load_puzzles, save_split, split_puzzles
from .checkpoints import file_sha256, save_training_state
from .env import legal_actions
from .io import append_jsonl, write_json, write_jsonl_gzip
from .method import Visit, compute_advantages


def _git_state() -> dict:
    def call(*args: str) -> str | None:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": call("rev-parse", "HEAD"), "status": call("status", "--short")}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "game24_experiment").glob("*.py"))
    files += [root / "game24_experiment" / "experiment.json",
              root / "game24_experiment" / "requirements.txt",
              root / "game24_experiment_protocol_and_logging.md",
              root / "two_step_gated_grpo_theory.md",
              root / "action_mean_grpo_baseline.md"]
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def _config_from_plan(plan: dict) -> ExperimentConfig:
    if plan.get("schema_version") != 2:
        raise ValueError("Recreate the experiment plan with the current prepare command")
    if plan.get("dataset_provenance") != dataset_provenance():
        raise ValueError("Plan does not pin the trusted official dataset")
    raw = dict(plan["config"])
    for key in ("training_seeds", "arms", "lora_target_modules"):
        raw[key] = tuple(raw[key])
    config = ExperimentConfig(**raw)
    config.validate()
    if len(config.model_revision) != 40 or any(c not in "0123456789abcdef" for c in config.model_revision):
        raise ValueError("Plan must pin a 40-character model revision SHA")
    if plan.get("tokenizer_revision") != config.model_revision:
        raise ValueError("Tokenizer must use the same locked repository revision")
    if not plan.get("special_tokens", {}).get("eos_token_ids"):
        raise ValueError("Plan must pin checkpoint EOS and padding token IDs")
    return config


def prepare(config_path: Path, output: Path) -> None:
    from huggingface_hub import HfApi
    from transformers import GenerationConfig
    from .modeling import checkpoint_special_tokens

    config = ExperimentConfig.from_json(config_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Plan directory is not empty: {output}")
    puzzles, data_hash = load_puzzles(Path(config.dataset_csv))
    split = split_puzzles(puzzles, data_hash, seed=config.split_seed)
    revision = (HfApi().model_info(config.model_id).sha
                if config.model_revision == "resolve_at_run_start_and_lock_sha"
                else config.model_revision)
    if revision is None or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Could not resolve immutable model revision")
    generation_config = GenerationConfig.from_pretrained(config.model_id, revision=revision)
    effective = config.record()
    effective["model_revision"] = revision
    plan = {
        "schema_version": 2,
        "created_utc": _utc_now(),
        "config": effective,
        "tokenizer_revision": revision,
        "special_tokens": checkpoint_special_tokens(generation_config),
        "dataset_provenance": dataset_provenance(),
        "split": split,
        "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "git": _git_state(),
        "source_hashes": _source_hashes(),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "plan.json", plan)
    save_split(output / "split.json", split)
    print(f"Prepared {output / 'plan.json'}; train={len(split['train'])}, "
          f"validation={len(split['validation'])}, test={len(split['test'])}")


def _flatten(trajectories: list[Any]) -> list[Any]:
    return [decision for trajectory in trajectories for decision in trajectory.decisions]


def _attach_old_and_reference_logprobs(runner: Any, trajectories: list[Any],
                                       score_batch_size: int) -> None:
    import torch
    from .modeling import score_logprobs

    decisions = _flatten(trajectories)
    runner.model.eval()
    for start in range(0, len(decisions), score_batch_size):
        chunk = decisions[start:start + score_batch_size]
        with torch.no_grad():
            old = score_logprobs(runner.model, chunk, pad_id=runner.pad_id,
                                  device=runner.device)
            with runner.model.disable_adapter():
                reference = score_logprobs(runner.model, chunk, pad_id=runner.pad_id,
                                           device=runner.device)
        for decision, old_row, ref_row in zip(chunk, old, reference):
            decision.old_logp = old_row.float().cpu().tolist()
            decision.reference_logp = ref_row.float().cpu().tolist()


def _visits(trajectories: list[Any]) -> list[Visit]:
    rows: list[Visit] = []
    for trajectory in trajectories:
        for step, decision in enumerate(trajectory.decisions):
            transition = decision.transition
            rows.append(Visit(
                decision.visit_id, trajectory.trajectory_id, step,
                transition.before, transition.action_key, transition.after,
                trajectory.reward,
            ))
    return rows


def _stats_record(trajectories: list[Any], advantage_batch: Any, *, arm: str) -> dict:
    from collections import Counter, defaultdict

    gates_by_state: dict = defaultdict(list)
    for gate in advantage_batch.gates.values():
        gates_by_state[gate.parent_state].append(gate)
    children: dict[tuple[str, ...], dict[str, tuple[str, ...] | None]] = {}
    for trajectory in trajectories:
        for decision in trajectory.decisions:
            transition = decision.transition
            children.setdefault(transition.before, {})[transition.action_key] = transition.after
    state_rows = []
    for state, info in advantage_batch.states.items():
        record = info.record()
        record["action_children"] = children.get(state, {})
        record["distinct_legal_actions_available"] = len(legal_actions(
            [Fraction(text) for text in state]
        ))
        gates = gates_by_state[state]
        positions = [advantage_batch.advantages[visit_id] for visit_id in info.visits]
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
                (row.final if arm == "two_step" else row.base) != 0 for row in positions
            ),
            "single_visit_action_count": sum(len(ids) == 1 for ids in info.action_visits.values()),
        }
        successors: dict = defaultdict(list)
        for action, child in children.get(state, {}).items():
            if child is not None:
                successors[child].append(action)
        record["same_successor_action_groups"] = [
            {"child_state": child, "actions": actions,
             "action_visit_counts": {action: len(info.action_visits[action]) for action in actions}}
            for child, actions in successors.items() if len(actions) > 1
        ]
        state_rows.append(record)
    return {
        "schema_version": 2,
        "arm": arm,
        "gate_usage": "applied" if arm == "two_step" else "counterfactual_diagnostic_only",
        "states": state_rows,
        "first_selected_visit_ids": sorted(advantage_batch.selected_first_visits),
        "gates": [gate.record() for gate in advantage_batch.gates.values()],
        "advantages": [
            {**row.record(), "applied_advantage": row.final if arm == "two_step" else row.base}
            for row in advantage_batch.advantages.values()
        ],
    }


def _diagnostic_metrics(trajectories: list[Any], advantage_batch: Any, *, arm: str) -> dict:
    from collections import Counter, defaultdict

    gate_reasons = Counter(gate.reason for gate in advantage_batch.gates.values())
    advantages = list(advantage_batch.advantages.values())
    same_child_groups = 0
    observed_legal_counts = []
    action_visit_counts = []
    for state in advantage_batch.states.values():
        observed_legal_counts.append(state.observed_legal_actions)
        action_visit_counts.extend(len(rows) for rows in state.action_visits.values())
    children: dict[tuple[str, ...], dict[tuple[str, ...], set[str]]] = defaultdict(lambda: defaultdict(set))
    for trajectory in trajectories:
        for decision in trajectory.decisions:
            transition = decision.transition
            if transition.after is not None:
                children[transition.before][transition.after].add(transition.action_key)
    same_child_groups = sum(len(actions) > 1 for by_child in children.values()
                            for actions in by_child.values())
    successful = sum(trajectory.reward for trajectory in trajectories)
    nonzero = sum((row.final if arm == "two_step" else row.base) != 0 for row in advantages)
    flags = []
    if successful == 0:
        flags.append("zero_success_batch")
    if nonzero == 0:
        flags.append("zero_policy_advantages")
    if arm == "two_step" and not gate_reasons["passed"]:
        flags.append("no_gate_passed")
    return {
        "diagnostic_flags": flags,
        "gate_usage": "applied" if arm == "two_step" else "counterfactual_diagnostic_only",
        "nonzero_applied_advantage_visit_count": nonzero,
        "nonzero_applied_advantage_visit_fraction": nonzero / len(advantages),
        "successful_trajectories": successful,
        "invalid_trajectories": sum(bool(trajectory.game.termination and
                                         trajectory.game.termination.startswith("invalid:"))
                                    for trajectory in trajectories),
        "state_count": len(advantage_batch.states),
        "zero_sigma_states": sum(state.sigma == 0 for state in advantage_batch.states.values()),
        "states_with_five_observed_legal_actions": sum(count >= 5 for count in observed_legal_counts),
        "observed_legal_action_counts": observed_legal_counts,
        "action_visit_counts": action_visit_counts,
        "candidate_gate_count": len(advantage_batch.gates),
        "gate_reasons": dict(gate_reasons),
        "passed_gate_count": gate_reasons["passed"],
        "first_selected_visit_count": len(advantage_batch.selected_first_visits),
        "lambda_eligible_visit_count": sum(row.B for row in advantages),
        "beta_eligible_visit_count": sum(row.P and row.selected_second for row in advantages),
        "overlap_eligible_visit_count": sum(row.overlap for row in advantages),
        "lambda_visit_count": sum(row.B for row in advantages) if arm == "two_step" else 0,
        "beta_selected_visit_count": sum(row.P and row.selected_second for row in advantages) if arm == "two_step" else 0,
        "overlap_visit_count": sum(row.overlap for row in advantages) if arm == "two_step" else 0,
        "same_successor_distinct_action_groups": same_child_groups,
        "selected_first_absolute_advantages": [
            abs(row.base) for row in advantages if row.selected_first
        ],
    }


def _update(model: Any, optimizer: Any, trajectories: list[Any], advantage_batch: Any,
            *, arm: str, config: ExperimentConfig, pad_id: int,
            device: Any, microbatch_size: int, token_log_path: Path) -> dict:
    import torch
    from .loss import decision_objective
    from .modeling import score_logprobs

    decisions = _flatten(trajectories)
    lengths = {trajectory.trajectory_id: len(trajectory.decisions) for trajectory in trajectories}
    trajectories_by_visit = {
        decision.visit_id: trajectory.trajectory_id
        for trajectory in trajectories for decision in trajectory.decisions
    }
    number_of_trajectories = len(trajectories)
    optimizer.zero_grad(set_to_none=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    weighted_objective = 0.0
    token_records: list[dict] = []
    clipped_count = 0
    token_count = 0
    kl_sum = 0.0
    for start in range(0, len(decisions), microbatch_size):
        chunk = decisions[start:start + microbatch_size]
        current_rows = score_logprobs(model, chunk, pad_id=pad_id, device=device)
        losses = []
        for decision, current in zip(chunk, current_rows):
            old = torch.tensor(decision.old_logp, dtype=torch.float32, device=device)
            reference = torch.tensor(decision.reference_logp, dtype=torch.float32, device=device)
            advantage = advantage_batch.advantages[decision.visit_id]
            coefficient = advantage.base if arm == "baseline" else advantage.final
            objective, token_info = decision_objective(
                current, old, reference, coefficient,
                epsilon=config.clip_epsilon, kappa=config.kl_coefficient,
            )
            trajectory_id = trajectories_by_visit[decision.visit_id]
            weight = 1.0 / (number_of_trajectories * lengths[trajectory_id])
            losses.append(-weight * objective)
            weighted_objective += weight * objective.detach().item()
            clipped_count += int(token_info["clipped"].sum().item())
            token_count += current.numel()
            kl_sum += float(token_info["kl"].sum().item())
            token_records.append({
                "visit_id": decision.visit_id,
                "trajectory_id": trajectory_id,
                "weight_after_token_mean": weight,
                "advantage": coefficient,
                "token_ids": decision.completion_ids,
                "old_logp": decision.old_logp,
                "reference_logp": decision.reference_logp,
                "current_logp": current.detach().float().cpu().tolist(),
                "ratio": token_info["ratio"].float().cpu().tolist(),
                "clipped_policy_term": token_info["policy"].float().cpu().tolist(),
                "kl": token_info["kl"].float().cpu().tolist(),
                "clipped_mask": token_info["clipped"].int().cpu().tolist(),
                "ratio_outside_interval_mask": token_info["ratio_outside_interval"].int().cpu().tolist(),
            })
        loss = torch.stack(losses).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss; no update or sample skip")
        loss.backward()
    grad_squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad_squared += float(parameter.grad.detach().float().square().sum().item())
    grad_norm = grad_squared ** 0.5
    if not grad_norm < float("inf"):
        raise FloatingPointError("Nonfinite gradient; no update or sample skip")
    write_jsonl_gzip(token_log_path, token_records)
    optimizer.step()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    return {
        "objective": weighted_objective,
        "loss": -weighted_objective,
        "gradient_norm": grad_norm,
        "generated_tokens": token_count,
        "clipped_token_count": clipped_count,
        "clipped_token_fraction": clipped_count / token_count,
        "mean_token_kl": kl_sum / token_count,
        "trajectory_count": number_of_trajectories,
        "decision_count": len(decisions),
        "learning_rate": config.learning_rate,
    }


def _evaluate(runner: Any, puzzles: list[Any], indices: list[int], *,
              label: str, output_path: Path) -> dict:
    selected = [(index, puzzles[index].numbers) for index in indices]
    trajectories = runner.rollout(selected, repetitions=1, label=label, do_sample=False)
    write_jsonl_gzip(output_path, (trajectory.record() for trajectory in trajectories))
    success = sum(trajectory.reward for trajectory in trajectories)
    invalid = sum(bool(trajectory.game.termination and
                       trajectory.game.termination.startswith("invalid:"))
                  for trajectory in trajectories)
    return {
        "successes": success,
        "puzzles": len(trajectories),
        "accuracy": success / len(trajectories),
        "invalid_trajectories": invalid,
        "model_calls": sum(len(trajectory.decisions) for trajectory in trajectories),
        "generated_tokens": sum(len(decision.completion_ids)
                                for trajectory in trajectories for decision in trajectory.decisions),
    }


def run(plan_path: Path, output: Path, *, arm: str, seed: int) -> None:
    import importlib.metadata
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .modeling import PolicyRunner

    if not torch.cuda.is_available():
        raise RuntimeError("Formal training requires the external A100 GPU")
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    config = _config_from_plan(plan)
    if arm not in config.arms or seed not in config.training_seeds:
        raise ValueError("Arm or seed is not in the locked experiment plan")
    if plan["source_hashes"] != _source_hashes():
        raise ValueError("Source files changed after the experiment plan was locked")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {output}")
    puzzles, data_hash = load_puzzles(Path(config.dataset_csv))
    if data_hash != plan["split"]["source_sha256"]:
        raise ValueError("Dataset does not match locked plan")
    if plan["split"] != split_puzzles(puzzles, data_hash, seed=config.split_seed):
        raise ValueError("Plan split does not match the confirmed deterministic split")
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_bytes(plan_bytes)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    manifest = {
        "schema_version": 2,
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

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=plan["tokenizer_revision"])
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Selected tokenizer has no EOS token")
    base = AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=config.model_revision, dtype=torch.bfloat16,
    ).to(device)
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    model.config.use_cache = True
    model.enable_input_require_grads()
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters")
    manifest["trainable_parameters"] = {"count": sum(parameter.numel() for _, parameter in trainable),
                                        "names": [name for name, _ in trainable]}
    manifest["generation_config_from_checkpoint"] = model.generation_config.to_dict()
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable], lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    runner = PolicyRunner(model, tokenizer, max_new_tokens=config.max_new_tokens,
                          generation_batch_size=config.generation_batch_size, device=device)
    if runner.special_tokens != plan["special_tokens"]:
        raise ValueError("Loaded generation EOS/padding tokens do not match the locked plan")
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
            runner, puzzles, plan["split"]["validation"],
            label=f"val:u{update}", output_path=output / f"validation_{update:04d}.jsonl.gz",
        )
        metrics.update({"update": update, "utc": _utc_now()})
        append_jsonl(output / "validations.jsonl", metrics)
        if metrics["accuracy"] > best_accuracy:  # earliest checkpoint wins ties
            best_accuracy = metrics["accuracy"]
            best_update = update
            best_weights = {name: parameter.detach().cpu().clone() for name, parameter in trainable}
            model.save_pretrained(output / "best_adapter")
            write_json(output / "best_selection.json", {
                "update": best_update, "validation_accuracy": best_accuracy,
                "audit_checkpoint": f"checkpoints/state_{best_update:04d}.pt",
                "rule": "highest validation greedy pass@1; earliest update on tie",
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
                      "best_update": best_update, "best_validation_accuracy": best_accuracy},
        )
        record["file"] = f"checkpoints/{record['file']}"
        append_jsonl(output / "checkpoints.jsonl", record)
        return record

    initial_checkpoint = checkpoint(0)
    expected_updates = config.epochs * ((len(ordered) + config.puzzles_per_update - 1) // config.puzzles_per_update)
    update_index = 0
    for epoch in range(config.epochs):
        if epoch:
            random.Random(seed + epoch).shuffle(ordered)
        for start in range(0, len(ordered), config.puzzles_per_update):
            update_index += 1
            indices = ordered[start:start + config.puzzles_per_update]
            selected = [(index, puzzles[index].numbers) for index in indices]
            trajectories = runner.rollout(
                selected, repetitions=config.trajectories_per_puzzle,
                label=f"train:{arm}:{seed}:u{update_index}", do_sample=True,
            )
            _attach_old_and_reference_logprobs(runner, trajectories, config.scoring_batch_size)
            visits = _visits(trajectories)
            advantage_batch = compute_advantages(
                visits, omega=config.omega, lambda_bonus=config.lambda_bonus,
                beta=config.beta, seed=seed + update_index,
            )
            rollout_path = output / f"rollouts_{update_index:04d}.jsonl.gz"
            stats_path = output / f"stats_{update_index:04d}.json"
            token_path = output / f"tokens_{update_index:04d}.jsonl.gz"
            write_jsonl_gzip(rollout_path, (trajectory.record() for trajectory in trajectories))
            write_json(stats_path, _stats_record(trajectories, advantage_batch, arm=arm))
            metrics = _update(
                model, optimizer, trajectories, advantage_batch,
                arm=arm, config=config, pad_id=runner.pad_id,
                device=device, microbatch_size=config.gradient_microbatch_size,
                token_log_path=token_path,
            )
            metrics["update_artifacts"] = {
                artifact.name: {"bytes": artifact.stat().st_size,
                                "sha256": file_sha256(artifact)}
                for artifact in (rollout_path, stats_path, token_path)
            }
            metrics.update(_diagnostic_metrics(trajectories, advantage_batch, arm=arm))
            metrics.update({"update": update_index, "epoch": epoch,
                            "puzzle_indices": indices, "utc": _utc_now(),
                            "old_policy_checkpoint": f"checkpoints/state_{update_index - 1:04d}.pt",
                            "new_policy_checkpoint": f"checkpoints/state_{update_index:04d}.pt"})
            if update_index % config.validation_every_updates == 0 or start + config.puzzles_per_update >= len(ordered):
                validation(update_index)
            checkpoint_record = checkpoint(update_index)
            metrics["new_checkpoint_sha256"] = checkpoint_record["sha256"]
            append_jsonl(output / "updates.jsonl", metrics)
            print(f"update={update_index}/{expected_updates} "
                  f"success={metrics['successful_trajectories']}/{len(trajectories)} "
                  f"gates={metrics['passed_gate_count']}/{metrics['candidate_gate_count']} "
                  f"lambda_applied={metrics['lambda_visit_count']} "
                  f"beta_applied={metrics['beta_selected_visit_count']} "
                  f"flags={metrics['diagnostic_flags']}", flush=True)
            if update_index == 1:
                manifest["estimated_audit_checkpoint_bytes"] = (
                    initial_checkpoint["bytes"] + expected_updates * checkpoint_record["bytes"]
                )
                write_json(output / "manifest.json", manifest)
    model.save_pretrained(output / "final_adapter")
    with torch.no_grad():
        for name, parameter in trainable:
            parameter.copy_(best_weights[name].to(device))
    test_metrics = _evaluate(
        runner, puzzles, plan["split"]["test"],
        label="test:selected", output_path=output / "test.jsonl.gz",
    )
    write_json(output / "test_metrics.json", {
        **test_metrics, "selected_update": best_update,
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
    prep = sub.add_parser("prepare", help="Lock model revision, data split, and config")
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    job = sub.add_parser("run", help="Run one arm and seed on the external GPU")
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
