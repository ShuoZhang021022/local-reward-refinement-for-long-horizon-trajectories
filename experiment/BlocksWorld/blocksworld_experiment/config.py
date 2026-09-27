"""Locked first-run choices for the six-block comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str
    model_revision: str
    dataset_dir: str
    dataset_size: int
    train_size: int
    validation_size: int
    test_size: int
    split_seed: int
    training_seeds: tuple[int, ...]
    arms: tuple[str, ...]
    max_steps: int
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
    problems_per_update: int
    trajectories_per_problem: int
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
    def from_json(cls, path: Path) -> ExperimentConfig:
        raw = json.loads(path.read_text(encoding="utf-8"))
        fields = set(cls.__dataclass_fields__)
        if set(raw) != fields:
            raise ValueError(f"Config missing={sorted(fields-set(raw))}, "
                             f"extra={sorted(set(raw)-fields)}")
        for key in ("training_seeds", "arms", "lora_target_modules"):
            raw[key] = tuple(raw[key])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.dataset_size != self.train_size + self.validation_size + self.test_size:
            raise ValueError("Split sizes must sum to the dataset size")
        if min(self.train_size, self.validation_size, self.test_size) < 1:
            raise ValueError("All splits must be nonempty")
        if self.max_steps != 16:
            raise ValueError("The confirmed Blocksworld action limit is 16")
        if set(self.arms) != {"baseline", "two_step"}:
            raise ValueError("Expected baseline and two_step arms")
        if len(set(self.training_seeds)) != len(self.training_seeds) or not self.training_seeds:
            raise ValueError("Training seeds must be distinct and nonempty")
        if self.reference_policy != "frozen_initial_model":
            raise ValueError("Unknown reference policy")
        if self.omega < 0 or self.lambda_bonus <= 0 or self.beta <= 1:
            raise ValueError("Invalid two-step coefficients")
        if self.clip_epsilon <= 0 or self.kl_coefficient <= 0:
            raise ValueError("Invalid objective coefficients")
        if self.optimizer != "AdamW" or self.learning_rate <= 0:
            raise ValueError("Unknown optimizer")
        if self.epochs < 1 or self.problems_per_update < 1 or self.trajectories_per_problem < 1:
            raise ValueError("Invalid training budget")
        if any(x < 1 for x in (self.generation_batch_size, self.scoring_batch_size,
                               self.gradient_microbatch_size, self.max_new_tokens,
                               self.validation_every_updates)):
            raise ValueError("Batch sizes and lengths must be positive")
        if self.rollout_temperature != 1 or self.rollout_top_p != 1 or self.rollout_top_k != 0:
            raise ValueError("Old-policy logprobs require raw-policy sampling")
        if self.evaluation_decoding != "greedy_single_attempt":
            raise ValueError("Unknown evaluation decoding")
        if self.lora_rank < 1 or self.lora_alpha < 1 or not self.lora_target_modules:
            raise ValueError("Missing LoRA scope")
        if self.lora_dropout != 0 or self.weight_decay != 0 or self.max_grad_norm is not None:
            raise ValueError("This first run has no added regularization or gradient clipping")

    def record(self) -> dict:
        return asdict(self)
