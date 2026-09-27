"""Single-node, eight-A100 synchronous HotpotQA training."""

from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
from typing import Any

from game24_experiment.checkpoints import _cpu_copy, file_sha256
from game24_experiment.io import append_jsonl, write_json, write_jsonl_gzip
from game24_experiment.method import compute_advantages
from game24_experiment.train import _attach_old_and_reference_logprobs, _update, _visits

from .data import load_train_and_dev, split_train_validation, validate_official_counts
from .train import (_config_from_plan, _diagnostic_metrics, _git_state,
                    _source_hashes, _stats_record, _trajectory_record_with_gate)


WORLD_SIZE = 8


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rank_seed(seed: int, rank: int) -> int:
    """Disjoint deterministic streams for the three confirmed experiment seeds."""
    if not 0 <= rank < WORLD_SIZE:
        raise ValueError("Rank must be between zero and seven")
    return seed * 1009 + rank


def _join_gzip_shards(destination: Path, shards: list[Path]) -> None:
    """Gzip permits concatenated members; stream without loading the audit into RAM."""
    with destination.open("xb") as target:
        for path in shards:
            with path.open("rb") as source:
                shutil.copyfileobj(source, target)
    for path in shards:
        path.unlink()


def _save_distributed_checkpoint(path: Path, model: Any, optimizer: Any,
                                 *, metadata: dict, rank: int, device: Any,
                                 dist: Any) -> dict | None:
    import torch

    local_rng = {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
                 "torch_cuda": torch.cuda.get_rng_state(device)}
    all_rng: list[dict] = [None] * WORLD_SIZE  # type: ignore[list-item]
    dist.all_gather_object(all_rng, local_rng)
    if rank != 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite audit checkpoint: {path}")
    payload = {
        "schema_version": 2,
        "metadata": metadata,
        "trainable_parameters": {
            name: _cpu_copy(parameter) for name, parameter in model.named_parameters()
            if parameter.requires_grad},
        "optimizer": _cpu_copy(optimizer.state_dict()),
        "rng_by_rank": all_rng,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {"file": path.name, "bytes": path.stat().st_size,
            "sha256": file_sha256(path), **metadata}


class SynchronizedOptimizer:
    """Average all eight local trajectory-mean gradients before AdamW.step()."""

    def __init__(self, optimizer: Any, parameters: list[Any], dist: Any):
        self.optimizer = optimizer
        self.parameters = parameters
        self.dist = dist
        self.gradient_norm: float | None = None

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        self.optimizer.zero_grad(*args, **kwargs)

    def step(self) -> None:
        import torch

        # Every rank has the same number of trajectories, including the last
        # partial question batch. Averaging local means is therefore the
        # global mean used by the original one-GPU objective.
        flat = torch.cat([
            (parameter.grad.detach().float().reshape(-1)
             if parameter.grad is not None
             else torch.zeros(parameter.numel(), dtype=torch.float32,
                              device=parameter.device))
            for parameter in self.parameters
        ])
        self.dist.all_reduce(flat, op=self.dist.ReduceOp.SUM)
        flat.div_(self.dist.get_world_size())
        self.gradient_norm = float(torch.linalg.vector_norm(flat).item())
        if not torch.isfinite(flat).all():
            raise FloatingPointError("Nonfinite synchronized gradient; no update")
        offset = 0
        for parameter in self.parameters:
            piece = flat[offset:offset + parameter.numel()].reshape(parameter.shape)
            if parameter.grad is None:
                parameter.grad = piece.to(dtype=parameter.dtype).clone()
            else:
                parameter.grad.copy_(piece.to(dtype=parameter.grad.dtype))
            offset += parameter.numel()
        self.optimizer.step()


def _distributed_evaluate(runner: Any, questions: dict[str, Any], ids: list[str],
                          *, label: str, output_path: Path, rank: int,
                          dist: Any) -> dict | None:
    if not ids:
        raise ValueError("Evaluation split cannot be empty")
    start = len(ids) * rank // WORLD_SIZE
    stop = len(ids) * (rank + 1) // WORLD_SIZE
    trajectories = []
    for position in range(start, stop, runner.generation_batch_size):
        selected = [(index, questions[ids[index]])
                    for index in range(position,
                                       min(position + runner.generation_batch_size, stop))]
        trajectories.extend(runner.rollout(selected, repetitions=1,
                                           label=label, do_sample=False))
    shard = output_path.with_name(f"{output_path.stem}.rank{rank}.jsonl.gz")
    write_jsonl_gzip(shard, (item.record() for item in trajectories))
    coverages = [item.support_document_coverage for item in trajectories
                 if item.support_document_coverage is not None]
    local = {
        "question_count": len(trajectories),
        "em_sum": sum(item.exact_match for item in trajectories),
        "f1_sum": sum((item.reward for item in trajectories), Fraction(0)),
        "read_count_sum": sum(len(item.session.read_order) for item in trajectories),
        "coverage_sum": sum(coverages, Fraction(0)),
        "coverage_count": len(coverages),
        "full_coverage_count": sum(value == 1 for value in coverages),
        "invalid_trajectories": sum(item.session.failed for item in trajectories),
        "model_calls": sum(item.record()["model_calls"] for item in trajectories),
        "generated_tokens": sum(item.record()["generated_tokens"] for item in trajectories),
    }
    parts: list[dict] = [None] * WORLD_SIZE  # type: ignore[list-item]
    dist.all_gather_object(parts, local)
    if rank != 0:
        return None
    total = lambda key: sum(part[key] for part in parts)
    count = total("question_count")
    if count != len(ids):
        raise RuntimeError("Distributed evaluation omitted questions")
    coverage_count = total("coverage_count")
    _join_gzip_shards(output_path, [output_path.with_name(
        f"{output_path.stem}.rank{index}.jsonl.gz") for index in range(WORLD_SIZE)])
    return {
        "question_count": count,
        "answer_em": total("em_sum") / count,
        "answer_f1": float(total("f1_sum") / count),
        "mean_read_count": total("read_count_sum") / count,
        "gold_support_document_coverage_mean": (
            float(total("coverage_sum") / coverage_count) if coverage_count else None),
        "full_gold_support_document_coverage_fraction": (
            total("full_coverage_count") / coverage_count if coverage_count else None),
        "invalid_trajectories": total("invalid_trajectories"),
        "model_calls": total("model_calls"),
        "generated_tokens": total("generated_tokens"),
    }


def _merge_update_metrics(parts: list[dict], gradient_norm: float) -> dict:
    count = sum(row["generated_tokens"] for row in parts)
    return {
        "objective": sum(row["objective"] for row in parts) / WORLD_SIZE,
        "loss": sum(row["loss"] for row in parts) / WORLD_SIZE,
        "gradient_norm": gradient_norm,
        "generated_tokens": count,
        "clipped_token_count": sum(row["clipped_token_count"] for row in parts),
        "clipped_token_fraction": sum(row["clipped_token_count"] for row in parts) / count,
        "mean_token_kl": sum(row["mean_token_kl"] * row["generated_tokens"]
                             for row in parts) / count,
        "trajectory_count": sum(row["trajectory_count"] for row in parts),
        "decision_count": sum(row["decision_count"] for row in parts),
        "learning_rate": parts[0]["learning_rate"],
        "token_audit_local_gradient_weight_to_global_factor": 1 / WORLD_SIZE,
        "per_rank_update_metrics": parts,
    }


def run(plan_path: Path, output: Path, *, arm: str, seed: int) -> None:
    import torch
    import torch.distributed as dist
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
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE or \
            int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != WORLD_SIZE:
        raise RuntimeError("Formal training requires torchrun with eight local CUDA processes")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not 0 <= rank < WORLD_SIZE or not 0 <= local_rank < WORLD_SIZE:
        raise RuntimeError("Expected single-node ranks zero through seven")
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        raise RuntimeError("Formal training requires eight visible CUDA GPUs")
    torch.cuda.set_device(local_rank)
    if any("A100" not in torch.cuda.get_device_name(index)
           for index in range(WORLD_SIZE)):
        raise RuntimeError("All eight visible GPUs must be A100")
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
    questions = {item.question_id: item for item in train + dev}
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            (output / "plan.json").write_bytes(plan_bytes)
        dist.barrier()
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
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
            "distributed": {"backend": "nccl", "single_node": True,
                            "world_size": WORLD_SIZE, "processes_per_gpu": 1,
                            "trajectories_per_question_global": config.trajectories_per_question,
                            "trajectories_per_question_per_rank":
                                config.trajectories_per_question // WORLD_SIZE,
                            "rank_seed_formula": "training_seed * 1009 + global_rank",
                            "rank_seeds": [rank_seed(seed, index)
                                           for index in range(WORLD_SIZE)],
                            "gradient_reduction": "mean of equal-size per-rank trajectory means",
                            "token_audit_weight_semantics":
                                "weight_after_token_mean is the local gradient weight; "
                                "multiply by 1/8 for the effective global weight"},
            "gpus": [torch.cuda.get_device_name(index) for index in range(WORLD_SIZE)],
            "gpu_total_memory": [torch.cuda.get_device_properties(index).total_memory
                                 for index in range(WORLD_SIZE)],
            "audit_checkpoint_policy": "initial_and_every_update_rank0_weights_optimizer_all_rank_rng",
        }
        if config.trajectories_per_question % WORLD_SIZE:
            raise ValueError("Global trajectories per question must divide by eight")
        if rank == 0:
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
        for _, parameter in trainable:
            dist.broadcast(parameter.data, src=0)
        optimizer = torch.optim.AdamW([parameter for _, parameter in trainable],
                                      lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        synchronized = SynchronizedOptimizer(
            optimizer, [parameter for _, parameter in trainable], dist)
        runner = PolicyRunner(model, tokenizer,
                              max_new_tokens=config.selection_max_new_tokens,
                              generation_batch_size=config.generation_batch_size,
                              device=device)
        if config.answer_max_new_tokens != runner.max_new_tokens:
            raise ValueError("Answer generation limit is not implemented separately")
        if runner.special_tokens != plan["special_tokens"]:
            raise ValueError("Checkpoint special tokens differ from the locked plan")
        if rank == 0:
            manifest["trainable_parameters"] = {
                "count": sum(parameter.numel() for _, parameter in trainable),
                "names": [name for name, _ in trainable]}
            manifest["generation_config_from_checkpoint"] = model.generation_config.to_dict()
            manifest["generation_overrides"] = {
                "training_selection_and_answer": runner.generation_options(do_sample=True),
                "evaluation_selection_and_answer": runner.generation_options(do_sample=False)}
            manifest["optimizer_effective_parameter_groups"] = [
                {key: value for key, value in group.items() if key != "params"}
                for group in optimizer.param_groups]
            write_json(output / "manifest.json", manifest)
        sampling_seed = rank_seed(seed, rank)
        random.seed(sampling_seed)
        torch.manual_seed(sampling_seed)
        torch.cuda.manual_seed(sampling_seed)

        best_f1 = -1.0
        best_update = -1
        best_weights: dict[str, Any] = {}

        def validation(update: int) -> None:
            nonlocal best_f1, best_update, best_weights
            metrics = _distributed_evaluate(
                runner, questions, split["validation_ids"], label=f"val:u{update}",
                output_path=output / f"validation_{update:05d}.jsonl.gz",
                rank=rank, dist=dist)
            if rank == 0:
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
                        "audit_checkpoint": f"checkpoints/state_{update:05d}.pt"})

        def checkpoint(update: int) -> dict | None:
            result = _save_distributed_checkpoint(
                output / "checkpoints" / f"state_{update:05d}.pt", model, optimizer,
                rank=rank, device=device, dist=dist,
                metadata={"update": update, "arm": arm, "seed": seed,
                          "plan_sha256": manifest["plan_sha256"],
                          "best_update": best_update,
                          "best_validation_answer_f1": best_f1})
            if rank == 0:
                result["file"] = f"checkpoints/{result['file']}"
                append_jsonl(output / "checkpoints.jsonl", result)
            return result

        validation(0)
        checkpoint(0)
        ordered = list(split["train_ids"])
        random.Random(seed).shuffle(ordered)
        if rank == 0:
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
                local_trajectories = runner.rollout(
                    selected, repetitions=config.trajectories_per_question // WORLD_SIZE,
                    label=f"train:{arm}:{seed}:u{update_index}:rank{rank}",
                    do_sample=True)
                _attach_old_and_reference_logprobs(
                    runner, local_trajectories, config.scoring_batch_size)
                gathered: list[list[Any]] | None = ([None] * WORLD_SIZE if rank == 0
                                                     else None)  # type: ignore[list-item]
                dist.gather_object(local_trajectories, gathered, dst=0)
                global_trajectories = ([item for shard in gathered for item in shard]
                                       if rank == 0 else None)
                batch = (compute_advantages(
                    _visits(global_trajectories), omega=config.omega,
                    lambda_bonus=config.lambda_bonus, beta=config.beta,
                    seed=seed + update_index) if rank == 0 else None)
                payload = [batch]
                dist.broadcast_object_list(payload, src=0)
                batch = payload[0]
                rollout_path = output / f"rollouts_{update_index:05d}.jsonl.gz"
                stats_path = output / f"stats_{update_index:05d}.json"
                token_path = output / f"tokens_{update_index:05d}.jsonl.gz"
                if rank == 0:
                    write_jsonl_gzip(rollout_path, (
                        _trajectory_record_with_gate(item, batch, arm=arm)
                        for item in global_trajectories))
                    write_json(stats_path, _stats_record(global_trajectories, batch,
                                                         arm=arm))
                token_shard = output / f"tokens_{update_index:05d}.rank{rank}.jsonl.gz"
                local_metrics = _update(
                    model, synchronized, local_trajectories, batch,
                    arm=arm, config=config, pad_id=runner.pad_id, device=device,
                    microbatch_size=config.gradient_microbatch_size,
                    token_log_path=token_shard)
                local_metrics["gradient_norm"] = synchronized.gradient_norm
                metrics_parts: list[dict] = [None] * WORLD_SIZE  # type: ignore[list-item]
                dist.all_gather_object(metrics_parts, local_metrics)
                if rank == 0:
                    _join_gzip_shards(token_path, [output /
                        f"tokens_{update_index:05d}.rank{index}.jsonl.gz"
                        for index in range(WORLD_SIZE)])
                    metrics = _merge_update_metrics(metrics_parts,
                                                    synchronized.gradient_norm)
                    if metrics["trajectory_count"] != (len(question_ids) *
                            config.trajectories_per_question):
                        raise RuntimeError("Distributed rollout count differs from locked budget")
                    metrics.update(_diagnostic_metrics(global_trajectories, batch, arm=arm))
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
                if rank == 0:
                    metrics["new_checkpoint_sha256"] = checkpoint_record["sha256"]
                    append_jsonl(output / "updates.jsonl", metrics)
                    print(f"update={update_index}/{expected_updates} "
                          f"answer_f1={metrics['answer_f1_mean']:.4f} "
                          f"gates={metrics['passed_gate_count']}/{metrics['candidate_gate_count']}",
                          flush=True)
        if rank == 0:
            model.save_pretrained(output / "final_adapter")
            with torch.no_grad():
                for name, parameter in trainable:
                    parameter.copy_(best_weights[name].to(device))
        for _, parameter in trainable:
            dist.broadcast(parameter.data, src=0)
        test_metrics = _distributed_evaluate(
            runner, questions, split["test_ids"], label="test:selected",
            output_path=output / "test.jsonl.gz", rank=rank, dist=dist)
        if rank == 0:
            write_json(output / "test_metrics.json", {
                **test_metrics, "selected_update": best_update,
                "selected_validation_answer_f1": best_f1,
                "selected_checkpoint": f"checkpoints/state_{best_update:05d}.pt"})
            manifest.update({"status": "complete", "finished_utc": _utc_now(),
                             "selected_update": best_update,
                             "completed_updates": update_index})
            write_json(output / "manifest.json", manifest)
        dist.barrier()
    finally:
        dist.destroy_process_group()
