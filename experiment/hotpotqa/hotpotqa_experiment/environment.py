"""Five-read interaction and history-free anchor grouping for HotpotQA.

The selection failure rule is terminal F1=0 without retry. Answer formatting
and generation-length rules are implemented by the caller after confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping

from game24_experiment.env import INVALID


MAX_READS = 5
PARAGRAPH_COUNT = 10


def title_metadata(question: "Question") -> dict[int, str]:
    """Expose only the original paragraph titles before they are read."""
    return {paragraph.paragraph_id: paragraph.title for paragraph in question.paragraphs}


def parse_selection(raw_output: str) -> tuple[str | None, int | None, str | None]:
    """Parse one structured selection without assigning a failure outcome."""
    try:
        result = json.loads(raw_output)
    except (TypeError, ValueError):
        return None, None, "invalid_json"
    if not isinstance(result, dict):
        return None, None, "not_an_object"
    if result == {"action": "submit"}:
        return "submit", None, None
    if (set(result) == {"action", "paragraph_id"}
            and result.get("action") == "read"
            and type(result.get("paragraph_id")) is int):
        return "read", result["paragraph_id"], None
    return None, None, "invalid_schema"


@dataclass(frozen=True)
class Paragraph:
    paragraph_id: int
    title: str
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    paragraphs: tuple[Paragraph, ...]
    gold_answer: str | None = None
    supporting_facts: tuple[tuple[str, int], ...] | None = None

    @classmethod
    def from_hotpot_record(cls, raw: Mapping[str, Any]) -> "Question":
        """Parse one official distractor record; keep gold fields out of prompts."""
        question_id = raw.get("_id")
        question_text = raw.get("question")
        context = raw.get("context")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("HotpotQA record requires a nonempty _id")
        if not isinstance(question_text, str) or not question_text:
            raise ValueError("HotpotQA record requires a nonempty question")
        if not isinstance(context, list) or len(context) != PARAGRAPH_COUNT:
            raise ValueError("This protocol requires exactly ten context paragraphs")

        paragraphs: list[Paragraph] = []
        for paragraph_id, item in enumerate(context, start=1):
            if (not isinstance(item, list) or len(item) != 2
                    or not isinstance(item[0], str) or not item[0]
                    or not isinstance(item[1], list)
                    or any(not isinstance(sentence, str) for sentence in item[1])):
                raise ValueError(f"Invalid paragraph {paragraph_id}")
            paragraphs.append(Paragraph(paragraph_id, item[0], tuple(item[1])))

        gold_answer = raw.get("answer")
        if gold_answer is not None and not isinstance(gold_answer, str):
            raise ValueError("answer must be text when present")
        raw_facts = raw.get("supporting_facts")
        facts = None
        if raw_facts is not None:
            if (not isinstance(raw_facts, list)
                    or any(not isinstance(fact, list) or len(fact) != 2
                           or not isinstance(fact[0], str)
                           or type(fact[1]) is not int for fact in raw_facts)):
                raise ValueError("Invalid supporting_facts")
            facts = tuple((fact[0], fact[1]) for fact in raw_facts)
        return cls(question_id, question_text, tuple(paragraphs), gold_answer, facts)


@dataclass(frozen=True)
class SelectionTransition:
    before: tuple[str, ...]
    action_key: str
    after: tuple[str, ...] | None
    available_actions: tuple[str, ...]
    raw_output: str
    failure_reason: str | None


@dataclass
class ReadingSession:
    question: Question
    unread_metadata: Mapping[int, str]
    read_order: list[int] = field(default_factory=list)
    answer_requested: bool = False
    final_answer: str | None = None
    termination: str | None = None
    answer_request_reason: str | None = None
    failed: bool = False

    def __post_init__(self) -> None:
        if set(self.unread_metadata) != set(range(1, PARAGRAPH_COUNT + 1)):
            raise ValueError("Explicit metadata is required for all ten paragraphs")
        if any(not isinstance(value, str) for value in self.unread_metadata.values()):
            raise ValueError("Each metadata value must be text")

    @property
    def read_ids(self) -> frozenset[int]:
        return frozenset(self.read_order)

    @property
    def unread_ids(self) -> tuple[int, ...]:
        return tuple(i for i in range(1, PARAGRAPH_COUNT + 1) if i not in self.read_ids)

    @property
    def selection_actions(self) -> tuple[str, ...]:
        """Return selectable action keys; forced answering has no selection action."""
        if self.answer_requested or self.final_answer is not None or self.failed:
            return ()
        return tuple(f"read:{i}" for i in self.unread_ids) + ("submit",)

    @property
    def anchor_key(self) -> tuple[str, ...] | None:
        """Unordered grouping key, separate from the model's ordered history."""
        if not self.selection_actions:
            return None
        return (
            f"question:{self.question.question_id}",
            *(f"read:{i}" for i in sorted(self.read_ids)),
            "UNREAD",
            *(f"{i}:{self.unread_metadata[i]}" for i in self.unread_ids),
        )

    def _history(self) -> list[dict[str, Any]]:
        history = []
        for step, paragraph_id in enumerate(self.read_order, start=1):
            paragraph = self.question.paragraphs[paragraph_id - 1]
            history.append({
                "step": step,
                "action": {"action": "read", "paragraph_id": paragraph_id},
                "observation": {
                    "paragraph_id": paragraph_id,
                    "title": paragraph.title,
                    "sentences": list(paragraph.sentences),
                },
            })
        return history

    def prompt_payload(self) -> dict[str, Any]:
        """Return visible information only, never gold answer or gold facts."""
        return {
            "question": self.question.text,
            "history": self._history(),
            "current_state": {
                "read_paragraph_ids": sorted(self.read_ids),
                "unread_paragraphs": [
                    {"paragraph_id": i, "metadata": self.unread_metadata[i]}
                    for i in self.unread_ids
                ],
                "reads_remaining": MAX_READS - len(self.read_order),
            },
        }

    def selection_prompt(self) -> str:
        if not self.selection_actions:
            raise RuntimeError("No selection decision is available")
        payload = self.prompt_payload()
        payload["available_actions"] = list(self.selection_actions)
        return (
            "Choose the next step for this question. The full text of an unread "
            "paragraph is hidden until it is read. Reply with exactly one JSON "
            'object: {"action":"read","paragraph_id":N} or {"action":"submit"}. '
            "Do not include an explanation or extra fields.\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )

    def answer_prompt(self) -> str:
        if not self.answer_requested or self.final_answer is not None:
            raise RuntimeError("An answer is not currently requested")
        payload = self.prompt_payload()
        if self.answer_request_reason == "early_submit":
            payload["history"].append({
                "step": len(self.read_order) + 1,
                "action": {"action": "submit"},
                "observation": {"answer_requested": True},
            })
        return (
            "Answer the question using the interaction history below. "
            "Output only the shortest answer phrase, or exactly yes/no for a "
            "yes-or-no question. Do not include an explanation, citation, or "
            "prefix.\n"
            + json.dumps(payload, ensure_ascii=False,
                         separators=(",", ":"))
        )

    def read(self, paragraph_id: int) -> None:
        """Execute a selected read. The fifth read requests a forced answer."""
        if self.answer_requested or self.final_answer is not None or self.failed:
            raise RuntimeError("No further paragraph selection is allowed")
        if type(paragraph_id) is not int or paragraph_id not in self.unread_ids:
            raise ValueError("Paragraph must be an unread numbered candidate")
        if len(self.read_order) >= MAX_READS:
            raise RuntimeError("Read budget exhausted")
        self.read_order.append(paragraph_id)
        if len(self.read_order) == MAX_READS:
            self.answer_requested = True
            self.termination = "read_limit"
            self.answer_request_reason = "read_limit"

    def submit(self) -> None:
        """Request an early answer after fewer than five reads."""
        if self.answer_requested or self.final_answer is not None or self.failed:
            raise RuntimeError("Answer has already been requested")
        self.answer_requested = True
        self.termination = "early_submit"
        self.answer_request_reason = "early_submit"

    def finish_answer(self, answer: str) -> None:
        if not self.answer_requested or self.final_answer is not None or self.failed:
            raise RuntimeError("An answer is not currently requested")
        if not isinstance(answer, str):
            raise TypeError("Final answer must be text")
        self.final_answer = answer

    def fail(self, reason: str) -> None:
        """Apply the confirmed terminal-zero rule to a model output failure."""
        if self.failed or self.final_answer is not None:
            raise RuntimeError("Trajectory already ended")
        if not reason:
            raise ValueError("A failure reason is required")
        self.failed = True
        self.answer_requested = False
        self.termination = f"invalid:{reason}"

    def apply_selection(self, raw_output: str) -> SelectionTransition:
        """Parse and execute one output; invalid actions terminate at F1=0."""
        before = self.anchor_key
        if before is None:
            raise RuntimeError("No selection decision is available")
        available = self.selection_actions
        action, paragraph_id, parse_error = parse_selection(raw_output)
        if parse_error is not None:
            self.fail(parse_error)
            return SelectionTransition(before, INVALID, None, available,
                                       raw_output, parse_error)
        if action == "submit":
            self.submit()
            return SelectionTransition(before, "submit", None, available,
                                       raw_output, None)
        assert action == "read" and paragraph_id is not None
        if paragraph_id not in self.unread_ids:
            reason = "invalid_paragraph_id"
            self.fail(reason)
            return SelectionTransition(before, INVALID, None, available,
                                       raw_output, reason)
        self.read(paragraph_id)
        return SelectionTransition(before, f"read:{paragraph_id}",
                                   self.anchor_key, available, raw_output, None)
