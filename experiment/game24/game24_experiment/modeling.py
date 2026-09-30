"""Model calls, token accounting, and stateless Game24 rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .env import Game24, Transition


SYSTEM_RULES = (
    "Solve the 24 game. You only see the numbers currently available. "
    "Choose two different numbered slots and one operator from + - * /. "
    "Reply with exactly three fields: left_slot right_slot operator. "
    "Do not give a result, explanation, or other text. "
    "Each response makes exactly one operation. Negative numbers, zero, and fractions are allowed; "
    "division by zero is forbidden. Aim to leave exactly 24 after all numbers have been used."
)


@dataclass
class Decision:
    visit_id: str
    prompt_text: str
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_tokens: list[str]
    ended_by_eos: bool
    transition: Transition
    old_logp: list[float] = field(default_factory=list)
    reference_logp: list[float] = field(default_factory=list)

    def record(self) -> dict:
        return {
            "visit_id": self.visit_id,
            "prompt_text": self.prompt_text,
            "prompt_ids": self.prompt_ids,
            "completion_ids": self.completion_ids,
            "completion_tokens": self.completion_tokens,
            "generated_token_mask": [1] * len(self.completion_ids),
            "ended_by_eos": self.ended_by_eos,
            "termination_token_id": self.completion_ids[-1] if self.ended_by_eos else None,
            "transition": asdict(self.transition),
            "old_logp": self.old_logp,
            "reference_logp": self.reference_logp,
        }


@dataclass
class Trajectory:
    trajectory_id: str
    puzzle_index: int
    numbers: tuple[int, int, int, int]
    game: Game24
    decisions: list[Decision] = field(default_factory=list)

    @property
    def reward(self) -> int:
        if not self.game.terminal or self.game.reward is None:
            raise RuntimeError("Incomplete trajectory")
        return self.game.reward

    def record(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "puzzle_index": self.puzzle_index,
            "numbers": self.numbers,
            "reward": self.reward,
            "termination": self.game.termination,
            "steps": len(self.decisions),
            "decisions": [decision.record() for decision in self.decisions],
            "final_expression": self.decisions[-1].transition.expression,
            "model_calls": len(self.decisions),
            "generated_tokens": sum(len(decision.completion_ids) for decision in self.decisions),
        }


def state_prompt_ids(tokenizer: Any, game: Game24) -> tuple[str, list[int]]:
    messages = [
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user", "content": f"Current numbers: {game.observation}"},
    ]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False,
    )
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if not ids or any(type(token) is not int for token in ids):
        raise TypeError("Chat template must return a nonempty list of token IDs")
    return tokenizer.decode(ids, skip_special_tokens=False), list(ids)


def checkpoint_special_tokens(generation_config: Any) -> dict:
    """Use the pinned checkpoint's full stop-token set; never narrow it silently."""
    eos = generation_config.eos_token_id
    eos_ids = [eos] if type(eos) is int else list(eos or [])
    if not eos_ids or any(type(token) is not int or token < 0 for token in eos_ids):
        raise ValueError("Checkpoint must specify nonnegative EOS token IDs")
    pad_id = generation_config.pad_token_id
    if type(pad_id) is not int or pad_id < 0:
        raise ValueError("Checkpoint must specify a padding token ID")
    return {"eos_token_ids": sorted(set(eos_ids)), "pad_token_id": pad_id}


class PolicyRunner:
    def __init__(self, model: Any, tokenizer: Any, *, max_new_tokens: int,
                 generation_batch_size: int, device: torch.device) -> None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires an EOS token")
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.generation_batch_size = generation_batch_size
        self.device = device
        self.special_tokens = checkpoint_special_tokens(model.generation_config)
        self.eos_ids = self.special_tokens["eos_token_ids"]
        self.pad_id = self.special_tokens["pad_token_id"]
        if tokenizer.eos_token_id not in self.eos_ids:
            raise ValueError("Tokenizer EOS disagrees with checkpoint generation configuration")

    def generation_options(self, *, do_sample: bool) -> dict:
        options = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "num_beams": 1,
            "eos_token_id": self.eos_ids,
            "pad_token_id": self.pad_id,
            "use_cache": True,
            "repetition_penalty": 1.0,
            "return_dict_in_generate": False,
            "forced_eos_token_id": None,
            "forced_bos_token_id": None,
        }
        if do_sample:
            options.update(temperature=1.0, top_p=1.0, top_k=0)
        return options

    def _generate(self, prompt_ids: list[list[int]], *, do_sample: bool) -> list[list[int]]:
        width = max(len(ids) for ids in prompt_ids)
        input_ids = torch.full(
            (len(prompt_ids), width), self.pad_id, dtype=torch.long, device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, ids in enumerate(prompt_ids):
            input_ids[row, width - len(ids):] = torch.tensor(ids, device=self.device)
            attention_mask[row, width - len(ids):] = 1
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **self.generation_options(do_sample=do_sample),
        }
        with torch.inference_mode():
            generated = self.model.generate(**kwargs)
        results: list[list[int]] = []
        for row in generated[:, width:].tolist():
            end = next((index + 1 for index, token in enumerate(row)
                        if token in self.eos_ids), len(row))
            results.append(row[:end])
        return results

    def rollout(self, indexed_puzzles: list[tuple[int, tuple[int, int, int, int]]],
                *, repetitions: int, label: str, do_sample: bool) -> list[Trajectory]:
        trajectories = [
            Trajectory(f"{label}:p{index}:r{repeat}", index, numbers, Game24(numbers))
            for index, numbers in indexed_puzzles for repeat in range(repetitions)
        ]
        self.model.eval()
        for _step in range(3):
            active = [trajectory for trajectory in trajectories if not trajectory.game.terminal]
            if not active:
                break
            for start in range(0, len(active), self.generation_batch_size):
                chunk = active[start:start + self.generation_batch_size]
                prompts = [state_prompt_ids(self.tokenizer, trajectory.game) for trajectory in chunk]
                generated = self._generate([ids for _, ids in prompts], do_sample=do_sample)
                for trajectory, (prompt_text, ids), completion in zip(chunk, prompts, generated):
                    ended = bool(completion and completion[-1] in self.eos_ids)
                    body = completion[:-1] if ended else completion
                    raw = self.tokenizer.decode(body, skip_special_tokens=False)
                    transition = (trajectory.game.step(raw) if ended else
                                  trajectory.game.fail(raw, "generation_limit"))
                    trajectory.decisions.append(Decision(
                        visit_id=f"{trajectory.trajectory_id}:{len(trajectory.decisions)}",
                        prompt_text=prompt_text,
                        prompt_ids=ids,
                        completion_ids=completion,
                        completion_tokens=self.tokenizer.convert_ids_to_tokens(completion),
                        ended_by_eos=ended,
                        transition=transition,
                    ))
        return trajectories


def score_logprobs(model: Any, decisions: list[Decision], *,
                   pad_id: int, device: torch.device) -> list[torch.Tensor]:
    """Score generated tokens under the selected adapter, preserving gradients."""
    if not decisions:
        return []
    full = [decision.prompt_ids + decision.completion_ids for decision in decisions]
    if any(not decision.prompt_ids or not decision.completion_ids for decision in decisions):
        raise ValueError("Prompt and completion must both contain tokens")
    width = max(map(len, full))
    input_ids = torch.full((len(full), width), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(full):
        input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row, :len(ids)] = 1
    logits = model(input_ids=input_ids, attention_mask=attention_mask,
                   use_cache=False).logits
    # Score every generated position in one tensor operation. Per-row softmax
    # launches one GPU kernel sequence per decision, which is costly for the
    # many short Game24 responses in each rollout batch.
    row_indices: list[int] = []
    logit_positions: list[int] = []
    lengths: list[int] = []
    for row, decision in enumerate(decisions):
        length = len(decision.completion_ids)
        lengths.append(length)
        row_indices.extend([row] * length)
        logit_positions.extend(range(len(decision.prompt_ids) - 1,
                                     len(decision.prompt_ids) + length - 1))
    rows = torch.tensor(row_indices, dtype=torch.long, device=device)
    positions = torch.tensor(logit_positions, dtype=torch.long, device=device)
    token_logits = logits[rows, positions].float()
    targets = input_ids[rows, positions + 1]
    logp = F.log_softmax(token_logits, dim=-1).gather(
        -1, targets.unsqueeze(-1),
    ).squeeze(-1)
    return list(logp.split(lengths))
