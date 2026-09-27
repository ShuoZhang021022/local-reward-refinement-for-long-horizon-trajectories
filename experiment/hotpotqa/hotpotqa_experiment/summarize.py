"""Pair completed HotpotQA arms on the same full distractor dev questions."""

from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import hashlib
import json
from math import isclose
from pathlib import Path

from game24_experiment.io import write_json

from .train import _config_from_plan


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


_GATE_COUNT_FIELDS = (
    "decision_visits", "selected_first_visits",
    "passing_selected_first_visits", "candidate_gate_actions",
    "nonterminal_candidate_gate_actions", "passing_candidate_gate_actions",
    "lambda_applied_visits", "beta_applied_visits",
)


def _aggregate_gate_by_step(updates: list[dict]) -> dict[str, dict]:
    totals = {str(step): {field: 0 for field in _GATE_COUNT_FIELDS}
              for step in range(1, 6)}
    for update in updates:
        by_step = update.get("gate_by_step")
        if not isinstance(by_step, dict) or set(by_step) != set(totals):
            raise ValueError("Update lacks the five-step gate breakdown")
        for step, counts in by_step.items():
            for field in _GATE_COUNT_FIELDS:
                value = counts.get(field)
                if type(value) is not int or value < 0:
                    raise ValueError(f"Invalid gate count at step {step}: {field}")
                totals[step][field] += value
        if (sum(by_step[str(step)]["candidate_gate_actions"] for step in range(1, 6))
                != update.get("candidate_gate_count")
                or sum(by_step[str(step)]["passing_candidate_gate_actions"]
                       for step in range(1, 6)) != update.get("passed_gate_count")
                or sum(by_step[str(step)]["lambda_applied_visits"]
                       for step in range(1, 6)) != update.get("lambda_applied_visits")
                or sum(by_step[str(step)]["beta_applied_visits"]
                       for step in range(1, 6)) != update.get("beta_applied_visits")):
            raise ValueError("Step gate totals disagree with update totals")
    for counts in totals.values():
        candidates = counts["candidate_gate_actions"]
        nonterminal = counts["nonterminal_candidate_gate_actions"]
        selected = counts["selected_first_visits"]
        passed = counts["passing_candidate_gate_actions"]
        counts["candidate_gate_pass_rate"] = passed / candidates if candidates else None
        counts["nonterminal_candidate_gate_pass_rate"] = (
            passed / nonterminal if nonterminal else None)
        counts["selected_first_visit_gate_pass_rate"] = (
            counts["passing_selected_first_visits"] / selected if selected else None)
    return totals


def _test_outcomes(path: Path, expected_ids: list[str]) -> tuple[dict[str, tuple[Fraction, int]], dict]:
    rows: dict[str, tuple[Fraction, int]] = {}
    calls = tokens = invalid = reads = 0
    support_coverages: list[float] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            question_id = row["question_id"]
            if question_id in rows:
                raise ValueError(f"Duplicate test question: {question_id}")
            f1 = Fraction(row["answer_f1_exact"])
            em = row["answer_em"]
            if not 0 <= f1 <= 1 or em not in (0, 1):
                raise ValueError(f"Invalid test score: {question_id}")
            if float(f1) != row["answer_f1"]:
                raise ValueError(f"Inconsistent F1 representation: {question_id}")
            if row["read_count"] != len(row["read_order"]) or row["read_count"] > 5:
                raise ValueError(f"Invalid read count: {question_id}")
            decisions = row["decisions"]
            answer = row["answer_generation"]
            expected_calls = len(decisions) + int(answer is not None)
            expected_tokens = sum(len(item["completion_ids"]) for item in decisions)
            if answer is not None:
                expected_tokens += len(answer["completion_ids"])
            if row["model_calls"] != expected_calls or row["generated_tokens"] != expected_tokens:
                raise ValueError(f"Inconsistent call/token counts: {question_id}")
            if str(row["termination"]).startswith("invalid:") and (f1 != 0 or em != 0):
                raise ValueError(f"Invalid trajectory has nonzero score: {question_id}")
            rows[question_id] = (f1, em)
            calls += expected_calls
            tokens += expected_tokens
            reads += row["read_count"]
            invalid += str(row["termination"]).startswith("invalid:")
            coverage = row.get("gold_support_document_coverage")
            if coverage is not None:
                if not isinstance(coverage, (int, float)) or not 0 <= coverage <= 1:
                    raise ValueError(f"Invalid support-document coverage: {question_id}")
                support_coverages.append(coverage)
    if set(rows) != set(expected_ids) or len(rows) != len(expected_ids):
        raise ValueError("Test trace does not cover the full planned dev split")
    metrics = {
        "question_count": len(rows),
        "answer_f1": float(sum((score for score, _ in rows.values()), Fraction(0))
                           / len(rows)),
        "answer_em": sum(em for _, em in rows.values()) / len(rows),
        "mean_read_count": reads / len(rows),
        "gold_support_document_coverage_mean": (
            sum(support_coverages) / len(support_coverages)
            if support_coverages else None),
        "full_gold_support_document_coverage_fraction": (
            sum(value == 1 for value in support_coverages) / len(support_coverages)
            if support_coverages else None),
        "model_calls": calls,
        "generated_tokens": tokens,
        "invalid_trajectories": invalid,
    }
    return rows, metrics


def _load_run(path: Path) -> tuple[dict, dict[str, tuple[Fraction, int]], dict]:
    manifest = _read_json(path / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"Run is not complete: {path}")
    plan_bytes = (path / "plan.json").read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != manifest["plan_sha256"]:
        raise ValueError(f"Locked plan hash differs: {path}")
    plan = json.loads(plan_bytes)
    if plan != manifest["plan"]:
        raise ValueError(f"Manifest plan differs: {path}")
    config = _config_from_plan(plan)
    if manifest["arm"] not in config.arms or manifest["seed"] not in config.training_seeds:
        raise ValueError(f"Unexpected arm or seed: {path}")
    updates_per_epoch = ((len(plan["split"]["train_ids"]) +
                          config.questions_per_update - 1) // config.questions_per_update)
    expected_updates = config.epochs * updates_per_epoch
    if manifest.get("completed_updates") != expected_updates:
        raise ValueError(f"Missing planned updates: {path}")
    updates = _read_jsonl(path / "updates.jsonl")
    if [row.get("update") for row in updates] != list(range(1, expected_updates + 1)):
        raise ValueError(f"Missing or unordered update records: {path}")
    training_gate_by_step = _aggregate_gate_by_step(updates)
    expected_validations = [0] + [
        update for update in range(1, expected_updates + 1)
        if update % config.validation_every_updates == 0
        or update % updates_per_epoch == 0
    ]
    validations = _read_jsonl(path / "validations.jsonl")
    if [row.get("update") for row in validations] != expected_validations:
        raise ValueError(f"Missing or unordered validations: {path}")
    best = max(validations, key=lambda row: row["answer_f1"])
    selected = _read_json(path / "best_selection.json")
    if (selected.get("update") != best["update"]
            or selected.get("validation_answer_f1") != best["answer_f1"]
            or selected.get("rule") != config.checkpoint_selection
            or selected.get("audit_checkpoint") !=
                f"checkpoints/state_{best['update']:05d}.pt"
            or manifest.get("selected_update") != best["update"]):
        raise ValueError(f"Selected checkpoint violates validation F1 rule: {path}")
    outcomes, metrics = _test_outcomes(path / "test.jsonl.gz",
                                       plan["split"]["test_ids"])
    recorded = _read_json(path / "test_metrics.json")
    for key, value in metrics.items():
        observed = recorded.get(key)
        agrees = (isclose(observed, value, rel_tol=0, abs_tol=1e-12)
                  if isinstance(value, float) and isinstance(observed, (int, float))
                  else observed == value)
        if not agrees:
            raise ValueError(f"Test metric {key} disagrees with trace: {path}")
    if (recorded.get("selected_update") != manifest.get("selected_update")
            or recorded.get("selected_validation_answer_f1") != best["answer_f1"]
            or recorded.get("selected_checkpoint") !=
                f"checkpoints/state_{best['update']:05d}.pt"):
        raise ValueError(f"Selected checkpoint mismatch: {path}")
    return manifest, outcomes, {**recorded, "training_gate_by_step": training_gate_by_step}


def summarize(run_directories: list[Path]) -> dict:
    if not run_directories:
        raise ValueError("Run directories are required")
    runs: dict[tuple[str, int], dict[str, tuple[Fraction, int]]] = {}
    run_metrics: dict[tuple[str, int], dict] = {}
    plans: set[str] = set()
    versions: list[dict] = []
    expected: set[tuple[str, int]] | None = None
    for path in run_directories:
        manifest, outcomes, metrics = _load_run(path)
        key = (manifest["arm"], manifest["seed"])
        if key in runs:
            raise ValueError(f"Duplicate run: {key}")
        plan = manifest["plan"]
        planned = {(arm, seed) for arm in plan["config"]["arms"]
                   for seed in plan["config"]["training_seeds"]}
        if expected is None:
            expected = planned
        elif planned != expected:
            raise ValueError("Run plans disagree on arms or seeds")
        runs[key] = outcomes
        run_metrics[key] = metrics
        plans.add(manifest["plan_sha256"])
        versions.append(manifest["versions"])
    if len(plans) != 1 or any(version != versions[0] for version in versions):
        raise ValueError("Paired runs require one plan and matching library versions")
    if set(runs) != expected:
        raise ValueError(f"Missing planned runs: {sorted((expected or set()) - set(runs))}")
    seeds = sorted(seed for arm, seed in runs if arm == "baseline")
    per_seed = []
    for seed in seeds:
        baseline = run_metrics[("baseline", seed)]
        two_step = run_metrics[("two_step", seed)]
        per_seed.append({
            "seed": seed,
            "baseline_answer_f1": baseline["answer_f1"],
            "two_step_answer_f1": two_step["answer_f1"],
            "answer_f1_difference": two_step["answer_f1"] - baseline["answer_f1"],
            "baseline_answer_em": baseline["answer_em"],
            "two_step_answer_em": two_step["answer_em"],
            "answer_em_difference": two_step["answer_em"] - baseline["answer_em"],
            "baseline_model_calls": baseline["model_calls"],
            "two_step_model_calls": two_step["model_calls"],
            "baseline_generated_tokens": baseline["generated_tokens"],
            "two_step_generated_tokens": two_step["generated_tokens"],
            "baseline_mean_read_count": baseline["mean_read_count"],
            "two_step_mean_read_count": two_step["mean_read_count"],
            "baseline_gold_support_document_coverage":
                baseline["gold_support_document_coverage_mean"],
            "two_step_gold_support_document_coverage":
                two_step["gold_support_document_coverage_mean"],
            "baseline_training_gate_by_step": baseline["training_gate_by_step"],
            "two_step_training_gate_by_step": two_step["training_gate_by_step"],
        })
    pooled_gate_by_step: dict[str, dict[str, dict]] = {}
    for arm in ("baseline", "two_step"):
        pooled_gate_by_step[arm] = {}
        for step in range(1, 6):
            step_key = str(step)
            counts = {
                field: sum(run_metrics[(arm, seed)]["training_gate_by_step"]
                           [step_key][field] for seed in seeds)
                for field in _GATE_COUNT_FIELDS
            }
            candidates = counts["candidate_gate_actions"]
            nonterminal = counts["nonterminal_candidate_gate_actions"]
            selected = counts["selected_first_visits"]
            passed = counts["passing_candidate_gate_actions"]
            counts["candidate_gate_pass_rate"] = (
                passed / candidates if candidates else None)
            counts["nonterminal_candidate_gate_pass_rate"] = (
                passed / nonterminal if nonterminal else None)
            counts["selected_first_visit_gate_pass_rate"] = (
                counts["passing_selected_first_visits"] / selected
                if selected else None)
            pooled_gate_by_step[arm][step_key] = counts
    question_ids = sorted(next(iter(runs.values())))
    per_question = [
        {
            "question_id": question_id,
            "mean_paired_f1_difference": float(sum((
                runs[("two_step", seed)][question_id][0]
                - runs[("baseline", seed)][question_id][0]
                for seed in seeds), Fraction(0)) / len(seeds)),
        }
        for question_id in question_ids
    ]
    return {
        "plan_sha256": next(iter(plans)),
        "test_questions": len(question_ids),
        "seeds": seeds,
        "per_seed": per_seed,
        "pooled_training_gate_by_step": pooled_gate_by_step,
        "mean_answer_f1_difference": sum(row["answer_f1_difference"]
                                         for row in per_seed) / len(per_seed),
        "mean_answer_em_difference": sum(row["answer_em_difference"]
                                         for row in per_seed) / len(per_seed),
        "per_question": per_question,
        "interpretation": "Descriptive paired comparison on the custom five-read variant; "
                          "no official HotpotQA joint score or causal attribution to reading alone.",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_json(args.output, summarize(args.run))


if __name__ == "__main__":
    main()
