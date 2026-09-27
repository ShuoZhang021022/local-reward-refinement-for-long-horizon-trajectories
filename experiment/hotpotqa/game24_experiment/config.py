"""Fully specified choices for the first Game24 training comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str
    model_revision: str
    dataset_csv: str
    split_seed: int
    training_seeds: tuple[int, ...]
    arms: tuple[str, ...]
    omega: float
    lambda_bonus: float
    beta: float
    clip_epsilon: float
    kl_coefficient: float
    reference_policy: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    learning_rate: float
    weight_decay: float
    optimizer: str
    epochs: int
    puzzles_per_update: int
    trajectories_per_puzzle: int
    generation_batch_size: int
    scoring_batch_size: int
    gradient_microbatch_size: int
    max_new_tokens: int
    rollout_temperature: float
    rollout_top_p: float
    rollout_top_k: int
    validation_every_updates: int
    evaluation_decoding: str
    max_grad_norm: float | None

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        fields = set(cls.__dataclass_fields__)
        if set(raw) != fields:
            raise ValueError(f"Config missing={sorted(fields-set(raw))}, extra={sorted(set(raw)-fields)}")
        for key in ("training_seeds", "arms", "lora_target_modules"):
            raw[key] = tuple(raw[key])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.model_id != "Qwen/Qwen3-4B-Instruct-2507":
            raise ValueError("Unexpected model for this experiment")
        if set(self.arms) != {"baseline", "two_step"}:
            raise ValueError("First experiment must compare baseline and two_step")
        if self.reference_policy != "frozen_initial_model":
            raise ValueError("Unknown reference policy")
        if self.omega < 0 or self.lambda_bonus <= 0 or self.beta <= 1:
            raise ValueError("Invalid two-step coefficients")
        if self.clip_epsilon <= 0 or self.kl_coefficient <= 0:
            raise ValueError("Clip and KL coefficients must be positive")
        if self.epochs < 1 or self.trajectories_per_puzzle < 1 or self.puzzles_per_update < 1:
            raise ValueError("Training budget must be positive")
        if any(value < 1 for value in (
            self.generation_batch_size, self.scoring_batch_size,
            self.gradient_microbatch_size, self.max_new_tokens,
            self.validation_every_updates,
        )):
            raise ValueError("Batch sizes, generation limit, and validation interval must be positive")
        if not self.training_seeds or len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("Training seeds must be a nonempty list of unique integers")
        if self.lora_rank < 1 or self.lora_alpha < 1 or not self.lora_target_modules:
            raise ValueError("LoRA scope must be specified")
        if self.rollout_temperature != 1 or self.rollout_top_p != 1 or self.rollout_top_k != 0:
            raise ValueError("First run uses raw-policy sampling for exact old-policy logprobs")
        if self.evaluation_decoding != "greedy_single_attempt":
            raise ValueError("Unknown evaluation protocol")
        if self.lora_dropout != 0 or self.weight_decay != 0 or self.max_grad_norm is not None:
            raise ValueError("Unexpected extra regularization or clipping")
        if self.optimizer != "AdamW" or self.learning_rate <= 0:
            raise ValueError("Unknown optimizer setting")

    def record(self) -> dict:
        return asdict(self)
