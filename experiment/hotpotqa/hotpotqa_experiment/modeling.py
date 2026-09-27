"""HotpotQA selective-reading rollouts using the shared Qwen policy runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any

import torch

from game24_experiment.env import INVALID
from game24_experiment.modeling import PolicyRunner as BasePolicyRunner

from .environment import (Question, ReadingSession, SelectionTransition,
                          title_metadata)
from .metrics import answer_em, answer_f1


def _prompt_ids(tokenizer: Any, content: str) -> tuple[str, list[int]]:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, return_dict=False,
    )
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if not ids or any(type(token) is not int for token in ids):
        raise TypeError("Chat template must yield a nonempty token ID list")
    return tokenizer.decode(ids, skip_special_tokens=False), list(ids)


@dataclass
class SelectionDecision:
    visit_id: str
    prompt_text: str
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_tokens: list[str]
    ended_by_eos: bool
    transition: SelectionTransition
    old_logp: list[float] = field(default_factory=list)
    reference_logp: list[float] = field(default_factory=list)

    def record(self) -> dict:
        return {
            "visit_id": self.visit_id,
            "prompt_text": self.prompt_text,
            "prompt_ids": self.prompt_ids,
            "completion_ids": self.completion_ids,
            "completion_tokens": self.completion_tokens,
            "ended_by_eos": self.ended_by_eos,
            "transition": asdict(self.transition),
            "old_logp": self.old_logp,
            "reference_logp": self.reference_logp,
        }


@dataclass
class AnswerGeneration:
    prompt_text: str
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_tokens: list[str]
    ended_by_eos: bool
    raw_output: str
    valid: bool

    def record(self) -> dict:
        return asdict(self)


@dataclass
class Trajectory:
    trajectory_id: str
    question: Question
    session: ReadingSession
    decisions: list[SelectionDecision] = field(default_factory=list)
    answer_generation: AnswerGeneration | None = None

    @property
    def reward(self) -> Fraction:
        if self.session.failed:
            return Fraction(0)
        if self.session.final_answer is None or self.question.gold_answer is None:
            raise RuntimeError("Incomplete or unlabeled trajectory")
        return answer_f1(self.session.final_answer, self.question.gold_answer)

    @property
    def exact_match(self) -> int:
        if self.session.failed:
            return 0
        if self.session.final_answer is None or self.question.gold_answer is None:
            raise RuntimeError("Incomplete or unlabeled trajectory")
        return answer_em(self.session.final_answer, self.question.gold_answer)

    @property
    def support_document_coverage(self) -> Fraction | None:
        read_titles = {self.question.paragraphs[i - 1].title
                       for i in self.session.read_order}
        gold_titles = ({title for title, _ in self.question.supporting_facts}
                       if self.question.supporting_facts is not None else set())
        return (Fraction(len(read_titles & gold_titles), len(gold_titles))
                if gold_titles else None)

    def record(self) -> dict:
        reward = self.reward
        support_coverage = self.support_document_coverage
        answer_calls = int(self.answer_generation is not None)
        answer_tokens = (len(self.answer_generation.completion_ids)
                         if self.answer_generation is not None else 0)
        return {
            "trajectory_id": self.trajectory_id,
            "question_id": self.question.question_id,
            "read_order": list(self.session.read_order),
            "read_count": len(self.session.read_order),
            "termination": self.session.termination,
            "answer_request_reason": self.session.answer_request_reason,
            "answer": self.session.final_answer,
            "answer_em": self.exact_match,
            "answer_f1": float(reward),
            "answer_f1_exact": str(reward),
            "gold_support_document_coverage": (float(support_coverage)
                                               if support_coverage is not None else None),
            "decisions": [item.record() for item in self.decisions],
            "answer_generation": (self.answer_generation.record()
                                  if self.answer_generation is not None else None),
            "model_calls": len(self.decisions) + answer_calls,
            "generated_tokens": sum(len(item.completion_ids) for item in self.decisions)
                                + answer_tokens,
            "policy_selection_tokens": sum(len(item.completion_ids)
                                           for item in self.decisions),
            "answer_tokens_excluded_from_policy_objective": answer_tokens,
        }


class PolicyRunner(BasePolicyRunner):
    """Generate selection decisions, then a separate untrained answer response."""

    def rollout(self, indexed_questions: list[tuple[int, Question]], *,
                repetitions: int, label: str, do_sample: bool) -> list[Trajectory]:
        if repetitions < 1:
            raise ValueError("Repetitions must be positive")
        trajectories = [
            Trajectory(f"{label}:q{index}:r{repeat}", question,
                       ReadingSession(question, title_metadata(question)))
            for index, question in indexed_questions for repeat in range(repetitions)
        ]
        self.model.eval()
        for _step in range(6):  # zero to five reads, then submit or forced answer
            active = [item for item in trajectories if item.session.selection_actions]
            if not active:
                break
            for start in range(0, len(active), self.generation_batch_size):
                chunk = active[start:start + self.generation_batch_size]
                prompts = [_prompt_ids(self.tokenizer, item.session.selection_prompt())
                           for item in chunk]
                completions = self._generate([ids for _, ids in prompts],
                                             do_sample=do_sample)
                for item, (prompt_text, prompt_ids), completion in zip(
                        chunk, prompts, completions):
                    ended = bool(completion and completion[-1] in self.eos_ids)
                    body = completion[:-1] if ended else completion
                    raw = self.tokenizer.decode(body, skip_special_tokens=False)
                    if ended:
                        transition = item.session.apply_selection(raw)
                    else:
                        before = item.session.anchor_key
                        if before is None:
                            raise RuntimeError("Missing selection anchor")
                        available = item.session.selection_actions
                        item.session.fail("selection_generation_limit")
                        transition = SelectionTransition(
                            before, INVALID, None, available, raw,
                            "selection_generation_limit",
                        )
                    item.decisions.append(SelectionDecision(
                        f"{item.trajectory_id}:{len(item.decisions)}",
                        prompt_text, prompt_ids, completion,
                        self.tokenizer.convert_ids_to_tokens(completion), ended,
                        transition,
                    ))
        if any(item.session.selection_actions for item in trajectories):
            raise RuntimeError("Selection loop exceeded the six-decision limit")

        answer_ready = [item for item in trajectories if item.session.answer_requested]
        for start in range(0, len(answer_ready), self.generation_batch_size):
            chunk = answer_ready[start:start + self.generation_batch_size]
            prompts = [_prompt_ids(self.tokenizer, item.session.answer_prompt())
                       for item in chunk]
            completions = self._generate([ids for _, ids in prompts],
                                         do_sample=do_sample)
            for item, (prompt_text, prompt_ids), completion in zip(
                    chunk, prompts, completions):
                ended = bool(completion and completion[-1] in self.eos_ids)
                body = completion[:-1] if ended else completion
                raw = self.tokenizer.decode(body, skip_special_tokens=False)
                valid = ended and bool(raw.strip())
                if valid:
                    item.session.finish_answer(raw.strip())
                else:
                    item.session.fail("answer_generation_limit" if not ended
                                      else "blank_answer")
                item.answer_generation = AnswerGeneration(
                    prompt_text, prompt_ids, completion,
                    self.tokenizer.convert_ids_to_tokens(completion), ended,
                    raw, valid,
                )
        return trajectories
