"""One PlanBench Blocksworld action per model response and environment step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from game24_experiment.modeling import Decision, PolicyRunner as TokenPolicyRunner

from .data import Instance
from .env import Blocksworld


SYSTEM_RULES = (
    "Solve a six-block Blocksworld-4ops task. Your hand holds at most one block. "
    "A block is clear when nothing is on it and it is not held. "
    "When the hand is empty, you may pick-up a clear block on the table, "
    "or unstack a clear block that is on another block. "
    "When holding a block, you may put it down on the table, "
    "or stack it on a clear block. Each response executes exactly one action. "
    "Reply with exactly one line in one of these forms: "
    "'pick-up a', 'put-down a', 'unstack a b', or 'stack a b'. "
    "In unstack a b, a is currently on b. In stack a b, put a on b. "
    "Use the current state and goal below; do not include reasoning or extra text. "
    "Achieve every goal fact within 16 valid actions."
)


def state_prompt_ids(tokenizer: Any, game: Blocksworld) -> tuple[str, list[int]]:
    messages = [
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user", "content": game.observation},
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


@dataclass
class Trajectory:
    trajectory_id: str
    problem_index: int
    instance: Instance
    game: Blocksworld
    decisions: list[Decision] = field(default_factory=list)

    @property
    def reward(self) -> int:
        if not self.game.terminal or self.game.reward is None:
            raise RuntimeError("Incomplete Blocksworld trajectory")
        return self.game.reward

    def record(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "problem_index": self.problem_index,
            "source_file": self.instance.filename,
            "source_sha256": self.instance.sha256,
            "shortest_plan_length": self.instance.shortest_length,
            "reward": self.reward,
            "termination": self.game.termination,
            "steps": len(self.decisions),
            "legal_actions_executed": self.game.steps,
            "decisions": [decision.record() for decision in self.decisions],
            "model_calls": len(self.decisions),
            "generated_tokens": sum(len(decision.completion_ids) for decision in self.decisions),
            "final_state": self.game.state_key,
        }


class PolicyRunner(TokenPolicyRunner):
    def __init__(self, *args: Any, max_steps: int = 16, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps

    def rollout(self, indexed_instances: list[tuple[int, Instance]], *,
                repetitions: int, label: str, do_sample: bool) -> list[Trajectory]:
        trajectories = [
            Trajectory(f"{label}:p{index}:r{repeat}", index, instance,
                       Blocksworld(instance.problem, max_steps=self.max_steps))
            for index, instance in indexed_instances for repeat in range(repetitions)
        ]
        self.model.eval()
        for _step in range(self.max_steps):
            active = [trajectory for trajectory in trajectories if not trajectory.game.terminal]
            if not active:
                break
            for start in range(0, len(active), self.generation_batch_size):
                chunk = active[start:start + self.generation_batch_size]
                prompts = [state_prompt_ids(self.tokenizer, trajectory.game)
                           for trajectory in chunk]
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
        if any(not trajectory.game.terminal for trajectory in trajectories):
            raise RuntimeError("Rollout ended before terminal transition")
        return trajectories
