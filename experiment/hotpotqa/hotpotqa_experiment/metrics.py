"""Exact rational form of the official HotpotQA answer-only EM and F1 rules."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import re
import string


def normalize_answer(value: str) -> str:
    lowered = value.lower()
    without_punctuation = "".join(char for char in lowered if char not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def answer_em(prediction: str, gold: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(gold))


def answer_f1(prediction: str, gold: str) -> Fraction:
    """Return exact token-overlap F1; float(value) matches the official score."""
    pred = normalize_answer(prediction)
    truth = normalize_answer(gold)
    special = {"yes", "no", "noanswer"}
    if (pred in special or truth in special) and pred != truth:
        return Fraction(0)
    pred_tokens, gold_tokens = pred.split(), truth.split()
    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return Fraction(0)
    return Fraction(2 * common, len(pred_tokens) + len(gold_tokens))
