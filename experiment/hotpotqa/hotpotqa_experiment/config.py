"""Explicitly recorded first-run HotpotQA comparison settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str
    model_revision: str
    train_json: str
    dev_json: str
    split_seed: int
    validation_fraction: float
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
    questions_per_update: int
    trajectories_per_question: int
    generation_batch_size: int
    scoring_batch_size: int
    gradient_microbatch_size: int
    selection_max_new_tokens: int
    answer_max_new_tokens: int
    rollout_temperature: float
    rollout_top_p: float
    rollout_top_k: int
    validation_every_updates: int
    evaluation_decoding: str
    checkpoint_selection: str
    max_grad_norm: float | None

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        fields = set(cls.__dataclass_fields__)
        if set(raw) != fields:
            raise ValueError(f"Missing={sorted(fields-set(raw))}; extra={sorted(set(raw)-fields)}")
        for key in ("training_seeds", "arms", "lora_target_modules"):
            raw[key] = tuple(raw[key])
        result = cls(**raw)
        result.validate()
        return result

    def validate(self) -> None:
        if self.model_id != "Qwen/Qwen3-4B-Instruct-2507":
            raise ValueError("Unexpected checkpoint")
        if self.split_seed != 20260926 or self.validation_fraction != 0.1:
            raise ValueError("Unexpected train/validation split")
        if self.training_seeds != (20260926, 20260927, 20260928):
            raise ValueError("Unexpected training seeds")
        if set(self.arms) != {"baseline", "two_step"}:
            raise ValueError("Both confirmed arms are required")
        if (self.omega, self.lambda_bonus, self.beta) != (1.0, 0.5, 1.5):
            raise ValueError("Unexpected method coefficients")
        if (self.clip_epsilon, self.kl_coefficient) != (0.2, 0.01):
            raise ValueError("Unexpected policy objective coefficients")
        if self.reference_policy != "frozen_initial_model":
            raise ValueError("Unexpected reference policy")
        if (self.lora_rank, self.lora_alpha, self.lora_dropout) != (16, 32, 0.0):
            raise ValueError("Unexpected LoRA settings")
        if self.lora_target_modules != (
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ):
            raise ValueError("Unexpected trainable parameter scope")
        if (self.learning_rate, self.weight_decay, self.optimizer) != (1e-5, 0.0, "AdamW"):
            raise ValueError("Unexpected optimizer")
        if (self.epochs, self.questions_per_update, self.trajectories_per_question,
                self.validation_every_updates) != (1, 8, 256, 10):
            raise ValueError("Unexpected training budget")
        if any(value < 1 for value in (self.generation_batch_size,
                                       self.scoring_batch_size,
                                       self.gradient_microbatch_size)):
            raise ValueError("Batch sizes must be positive")
        if (self.selection_max_new_tokens, self.answer_max_new_tokens) != (32, 32):
            raise ValueError("Unexpected generation length")
        if (self.rollout_temperature, self.rollout_top_p, self.rollout_top_k) != (1.0, 1.0, 0):
            raise ValueError("Expected raw-policy training sampling")
        if self.evaluation_decoding != "greedy_single_attempt":
            raise ValueError("Unexpected evaluation decoding")
        if self.checkpoint_selection != "highest_validation_answer_f1_earliest_tie":
            raise ValueError("Unexpected checkpoint rule")
        if self.max_grad_norm is not None:
            raise ValueError("No extra gradient clipping was authorized")
        if not self.train_json or not self.dev_json:
            raise ValueError("Both labeled dataset paths are required")

    def record(self) -> dict:
        return asdict(self)
