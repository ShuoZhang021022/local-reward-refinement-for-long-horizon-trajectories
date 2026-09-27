"""Load labeled HotpotQA distractor files without deciding the train split."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from random import Random

from .environment import Question


OFFICIAL_TRAIN_COUNT = 90447
OFFICIAL_DISTRACTOR_DEV_COUNT = 7405


def validate_official_counts(train: list[Question], dev: list[Question]) -> None:
    """Reject partial files before a run can claim full official split roles."""
    if (len(train) != OFFICIAL_TRAIN_COUNT or
            len(dev) != OFFICIAL_DISTRACTOR_DEV_COUNT):
        raise ValueError(
            f"Expected full HotpotQA distractor files with "
            f"{OFFICIAL_TRAIN_COUNT} train and {OFFICIAL_DISTRACTOR_DEV_COUNT} dev "
            f"questions; got {len(train)} and {len(dev)}"
        )


def load_labeled_file(path: Path) -> tuple[list[Question], str]:
    """Require every example to have ten paragraphs and a gold answer."""
    source = path.read_bytes()
    raw = json.loads(source)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Expected a nonempty JSON array: {path}")
    questions = [Question.from_hotpot_record(item) for item in raw]
    ids = [item.question_id for item in questions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate question IDs in {path}")
    if any(question.gold_answer is None or question.supporting_facts is None
           for question in questions):
        raise ValueError(f"Labeled answer and supporting facts required: {path}")
    return questions, sha256(source).hexdigest()


def load_train_and_dev(train_path: Path, dev_path: Path) -> tuple[
    list[Question], list[Question], dict[str, str]
]:
    """Keep official dev separate so it can be used only as final test."""
    train, train_hash = load_labeled_file(train_path)
    dev, dev_hash = load_labeled_file(dev_path)
    overlap = {item.question_id for item in train} & {item.question_id for item in dev}
    if overlap:
        raise ValueError(f"Train/dev question ID overlap: {sorted(overlap)[:3]}")
    return train, dev, {"train_sha256": train_hash, "dev_sha256": dev_hash}


def split_train_validation(train: list[Question], dev: list[Question],
                           *, seed: int, validation_fraction: float) -> dict[str, object]:
    """Deterministically reserve 10% of official train; preserve all of dev."""
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    if len(train) < 2 or not dev:
        raise ValueError("Need at least two train and one dev question")
    if len({item.question_id for item in train + dev}) != len(train) + len(dev):
        raise ValueError("Question IDs must be unique across train and dev")
    positions = list(range(len(train)))
    Random(seed).shuffle(positions)
    validation_count = round(len(train) * validation_fraction)
    if not 0 < validation_count < len(train):
        raise ValueError("Validation split is empty or consumes all train data")
    return {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "validation_count_rounding": "Python round to nearest integer, ties to even",
        "train_ids": [train[index].question_id for index in positions[validation_count:]],
        "validation_ids": [train[index].question_id for index in positions[:validation_count]],
        "test_ids": [item.question_id for item in dev],
    }
