from __future__ import annotations

from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from game24_experiment.env import INVALID
from game24_experiment.method import Visit, compute_advantages
from hotpotqa_experiment.config import ExperimentConfig
from hotpotqa_experiment.data import (load_labeled_file, load_train_and_dev,
                                      split_train_validation,
                                      validate_official_counts)
from hotpotqa_experiment.environment import (
    Question, ReadingSession, parse_selection, title_metadata,
)
from hotpotqa_experiment.metrics import answer_em, answer_f1
from hotpotqa_experiment.modeling import PolicyRunner
from hotpotqa_experiment.train import (_gate_by_step, _trajectory_record_with_gate,
                                       prepare, run)
from hotpotqa_experiment.summarize import summarize


def fixture_question() -> Question:
    raw = {
        "_id": "question-1",
        "question": "Where was the researcher born?",
        "answer": "Gold Hidden Answer",
        "supporting_facts": [["Article 1", 0]],
        "context": [
            [f"Article {i}", [f"Hidden paragraph {i} sentence one.",
                             f"Hidden paragraph {i} sentence two."]]
            for i in range(1, 11)
        ],
    }
    return Question.from_hotpot_record(raw)


class HotpotEnvironmentTests(unittest.TestCase):
    def test_same_unordered_anchor_keeps_distinct_full_histories(self) -> None:
        question = fixture_question()
        one = ReadingSession(question, title_metadata(question))
        two = ReadingSession(question, title_metadata(question))
        one.read(1)
        one.read(2)
        two.read(2)
        two.read(1)
        self.assertEqual(one.anchor_key, two.anchor_key)
        self.assertNotEqual(one.selection_prompt(), two.selection_prompt())
        self.assertEqual(one.selection_actions, two.selection_actions)

    def test_unread_text_and_gold_stay_hidden_until_read(self) -> None:
        question = fixture_question()
        session = ReadingSession(question, title_metadata(question))
        before = session.selection_prompt()
        self.assertIn("Article 7", before)
        self.assertNotIn("Hidden paragraph 7", before)
        self.assertNotIn("Gold Hidden Answer", before)
        self.assertNotIn("supporting_facts", before)
        session.read(7)
        self.assertIn("Hidden paragraph 7", session.selection_prompt())
        self.assertNotIn("Hidden paragraph 8", session.selection_prompt())

    def test_five_read_limit_and_early_submit(self) -> None:
        question = fixture_question()
        session = ReadingSession(question, title_metadata(question))
        self.assertEqual(len(session.selection_actions), 11)
        for i in range(1, 5):
            session.read(i)
        self.assertEqual(len(session.selection_actions), 7)
        self.assertIsNotNone(session.anchor_key)
        session.read(5)
        self.assertEqual(session.selection_actions, ())
        self.assertIsNone(session.anchor_key)
        self.assertEqual(session.termination, "read_limit")
        self.assertNotIn("Gold Hidden Answer", session.answer_prompt())
        session.finish_answer("a city")
        with self.assertRaises(RuntimeError):
            session.read(6)

        early = ReadingSession(question, title_metadata(question))
        early.read(2)
        early.submit()
        self.assertEqual(early.termination, "early_submit")
        self.assertEqual(early.selection_actions, ())
        self.assertIsNone(early.anchor_key)
        self.assertIn("Hidden paragraph 2", early.answer_prompt())

    def test_answer_history_includes_early_submit_control_action(self) -> None:
        question = fixture_question()
        for read_ids in ((), (2, 4)):
            session = ReadingSession(question, title_metadata(question))
            for paragraph_id in read_ids:
                session.read(paragraph_id)
            session.submit()
            payload = json.loads(session.answer_prompt().split("\n", 1)[1])
            self.assertEqual(len(payload["history"]), len(read_ids) + 1)
            self.assertEqual(payload["history"][-1], {
                "step": len(read_ids) + 1,
                "action": {"action": "submit"},
                "observation": {"answer_requested": True},
            })

        forced = ReadingSession(question, title_metadata(question))
        for paragraph_id in range(1, 6):
            forced.read(paragraph_id)
        forced_payload = json.loads(forced.answer_prompt().split("\n", 1)[1])
        self.assertEqual(len(forced_payload["history"]), 5)
        self.assertTrue(all(item["action"]["action"] == "read"
                            for item in forced_payload["history"]))

    def test_selection_parser_accepts_one_structured_action(self) -> None:
        self.assertEqual(parse_selection('{"action":"read","paragraph_id":3}'),
                         ("read", 3, None))
        self.assertEqual(parse_selection('{"action":"submit"}'),
                         ("submit", None, None))
        self.assertEqual(parse_selection('{"action":"read","paragraph_id":true}')[2],
                         "invalid_schema")
        self.assertEqual(parse_selection("not json")[2], "invalid_json")

    def test_invalid_selection_is_terminal_with_zero_reward_action(self) -> None:
        question = fixture_question()
        session = ReadingSession(question, title_metadata(question))
        session.read(1)
        transition = session.apply_selection('{"action":"read","paragraph_id":1}')
        self.assertEqual(transition.action_key, INVALID)
        self.assertEqual(transition.failure_reason, "invalid_paragraph_id")
        self.assertIsNone(transition.after)
        self.assertTrue(session.failed)
        self.assertEqual(session.selection_actions, ())
        self.assertEqual(session.termination, "invalid:invalid_paragraph_id")
        with self.assertRaises(RuntimeError):
            session.apply_selection('{"action":"submit"}')

    def test_valid_selection_child_anchor_and_forced_answer(self) -> None:
        question = fixture_question()
        session = ReadingSession(question, title_metadata(question))
        for paragraph_id in range(1, 6):
            prior = session.anchor_key
            transition = session.apply_selection(
                json.dumps({"action": "read", "paragraph_id": paragraph_id})
            )
            self.assertEqual(transition.before, prior)
            self.assertEqual(transition.action_key, f"read:{paragraph_id}")
            self.assertEqual(transition.after, session.anchor_key)
        self.assertIsNone(transition.after)
        self.assertTrue(session.answer_requested)

    def test_labeled_dataset_loader_rejects_overlap(self) -> None:
        raw = {"_id": "question-1", "question": "Q?", "answer": "A",
               "supporting_facts": [["Article 1", 0]],
               "context": [[f"Article {i}", ["Sentence."]] for i in range(1, 11)]}
        with TemporaryDirectory() as root:
            train_path = Path(root) / "train.json"
            dev_path = Path(root) / "dev.json"
            train_path.write_text(json.dumps([raw]), encoding="utf-8")
            dev_path.write_text(json.dumps([raw]), encoding="utf-8")
            questions, source_hash = load_labeled_file(train_path)
            self.assertEqual(len(questions), 1)
            self.assertEqual(len(source_hash), 64)
            with self.assertRaisesRegex(ValueError, "full HotpotQA"):
                validate_official_counts(questions, questions)
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_train_and_dev(train_path, dev_path)

    def test_confirmed_config_and_disjoint_split(self) -> None:
        config = ExperimentConfig.from_json(
            Path(__file__).resolve().parents[1] / "hotpotqa_experiment" /
            "experiment.json")
        self.assertEqual(config.trajectories_per_question, 256)
        train = [Question(f"t{i}", "Q", fixture_question().paragraphs, "A", ())
                 for i in range(20)]
        dev = [Question(f"d{i}", "Q", fixture_question().paragraphs, "A", ())
               for i in range(3)]
        split = split_train_validation(train, dev, seed=20260926,
                                       validation_fraction=0.1)
        self.assertEqual(len(split["validation_ids"]), 2)
        self.assertEqual(len(split["train_ids"]), 18)
        self.assertEqual(split["test_ids"], ["d0", "d1", "d2"])
        self.assertEqual(len(set(split["train_ids"] + split["validation_ids"] +
                                 split["test_ids"])), 23)

    def test_runner_separates_answer_tokens_and_scores_f1(self) -> None:
        class CharacterTokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, tokenize,
                                    add_generation_prompt, return_dict):
                return [ord(char) + 1 for char in messages[-1]["content"]]

            def decode(self, ids, skip_special_tokens):
                return "".join(chr(token - 1) for token in ids if token != 0)

            def convert_ids_to_tokens(self, ids):
                return [str(token) for token in ids]

        class ScriptedModel:
            generation_config = SimpleNamespace(eos_token_id=0, pad_token_id=1)

            def __init__(self):
                self.calls = 0

            def eval(self):
                return self

            def generate(self, **kwargs):
                self.calls += 1
                command = [
                    '{"action":"read","paragraph_id":1}',
                    '{"action":"submit"}',
                    "Gold Hidden Answer",
                ][self.calls - 1]
                tail = torch.tensor([[ord(char) + 1 for char in command] + [0]])
                return torch.cat([kwargs["input_ids"], tail], dim=1)

        model = ScriptedModel()
        runner = PolicyRunner(model, CharacterTokenizer(), max_new_tokens=32,
                              generation_batch_size=1, device=torch.device("cpu"))
        trajectory = runner.rollout([(0, fixture_question())], repetitions=1,
                                    label="test", do_sample=False)[0]
        self.assertEqual(trajectory.reward, 1)
        self.assertEqual(trajectory.exact_match, 1)
        self.assertEqual(trajectory.session.read_order, [1])
        self.assertEqual(model.calls, 3)
        self.assertEqual(len(trajectory.decisions), 2)
        self.assertIsNotNone(trajectory.answer_generation)
        self.assertEqual(trajectory.record()["model_calls"], 3)
        self.assertEqual(trajectory.record()["answer_tokens_excluded_from_policy_objective"],
                         len(trajectory.answer_generation.completion_ids))

    def test_missing_eos_fails_without_followup_answer_call(self) -> None:
        class CharacterTokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, tokenize,
                                    add_generation_prompt, return_dict):
                return [ord(char) + 1 for char in messages[-1]["content"]]

            def decode(self, ids, skip_special_tokens):
                return "".join(chr(token - 1) for token in ids if token != 0)

            def convert_ids_to_tokens(self, ids):
                return [str(token) for token in ids]

        class NoEosModel:
            generation_config = SimpleNamespace(eos_token_id=0, pad_token_id=1)

            def __init__(self):
                self.calls = 0

            def eval(self):
                return self

            def generate(self, **kwargs):
                self.calls += 1
                output = '{"action":"submit"}'
                return torch.cat([kwargs["input_ids"],
                                  torch.tensor([[ord(char) + 1 for char in output]])],
                                 dim=1)

        model = NoEosModel()
        runner = PolicyRunner(model, CharacterTokenizer(), max_new_tokens=32,
                              generation_batch_size=1, device=torch.device("cpu"))
        trajectory = runner.rollout([(0, fixture_question())], repetitions=1,
                                    label="no-eos", do_sample=False)[0]
        self.assertEqual(model.calls, 1)
        self.assertEqual(trajectory.reward, 0)
        self.assertEqual(trajectory.session.termination,
                         "invalid:selection_generation_limit")
        self.assertIsNone(trajectory.answer_generation)

    def test_prepare_locks_synthetic_data_and_run_refuses_cpu(self) -> None:
        raw = {"question": "Q?", "answer": "A",
               "supporting_facts": [["Article 1", 0]],
               "context": [[f"Article {i}", ["Sentence."]] for i in range(1, 11)]}
        with TemporaryDirectory() as root:
            directory = Path(root)
            train_path = directory / "train.json"
            dev_path = directory / "dev.json"
            train_path.write_text(json.dumps([{**raw, "_id": f"t{i}"}
                                              for i in range(20)]), encoding="utf-8")
            dev_path.write_text(json.dumps([{**raw, "_id": f"d{i}"}
                                            for i in range(3)]), encoding="utf-8")
            source = Path(__file__).resolve().parents[1] / "hotpotqa_experiment" / "experiment.json"
            config = json.loads(source.read_text(encoding="utf-8"))
            config["train_json"] = str(train_path)
            config["dev_json"] = str(dev_path)
            config["model_revision"] = "a" * 40
            config_path = directory / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            plan_dir = directory / "plan"
            with patch("transformers.GenerationConfig.from_pretrained",
                       return_value=SimpleNamespace(eos_token_id=0, pad_token_id=1)), \
                    patch("hotpotqa_experiment.data.OFFICIAL_TRAIN_COUNT", 20), \
                    patch("hotpotqa_experiment.data.OFFICIAL_DISTRACTOR_DEV_COUNT", 3):
                prepare(config_path, plan_dir)
            plan = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(len(plan["split"]["train_ids"]), 18)
            self.assertEqual(len(plan["split"]["validation_ids"]), 2)
            self.assertEqual(len(plan["split"]["test_ids"]), 3)
            with patch("torch.cuda.is_available", return_value=False), \
                    patch("hotpotqa_experiment.data.OFFICIAL_TRAIN_COUNT", 20), \
                    patch("hotpotqa_experiment.data.OFFICIAL_DISTRACTOR_DEV_COUNT", 3):
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    run(plan_dir / "plan.json", directory / "run",
                        arm="baseline", seed=20260926)
            self.assertFalse((directory / "run").exists())

    def test_summary_pairs_all_six_runs_on_same_dev_questions(self) -> None:
        source = Path(__file__).resolve().parents[1] / "hotpotqa_experiment" / "experiment.json"
        config = json.loads(source.read_text(encoding="utf-8"))
        config["model_revision"] = "a" * 40
        plan = {"schema_version": 1, "config": config,
                "tokenizer_revision": "a" * 40,
                "special_tokens": {"eos_token_ids": [0], "pad_token_id": 1},
                "split": {"train_ids": [f"t{i}" for i in range(18)],
                          "validation_ids": ["v0", "v1"],
                          "test_ids": ["d0", "d1"]}}
        plan_bytes = json.dumps(plan).encode("utf-8")
        plan_hash = hashlib.sha256(plan_bytes).hexdigest()
        with TemporaryDirectory() as root:
            paths = []
            for arm in ("baseline", "two_step"):
                for seed in config["training_seeds"]:
                    directory = Path(root) / f"{arm}_{seed}"
                    directory.mkdir()
                    paths.append(directory)
                    (directory / "plan.json").write_bytes(plan_bytes)
                    (directory / "manifest.json").write_text(json.dumps({
                        "status": "complete", "arm": arm, "seed": seed,
                        "plan_sha256": plan_hash, "plan": plan,
                        "completed_updates": 3, "selected_update": 3,
                        "versions": {"torch": "fixture"},
                    }), encoding="utf-8")
                    score = 1 if arm == "two_step" else 0
                    gate_steps = {str(step): {
                        "decision_visits": int(step == 1),
                        "selected_first_visits": int(step == 1),
                        "passing_selected_first_visits": score if step == 1 else 0,
                        "candidate_gate_actions": int(step == 1),
                        "nonterminal_candidate_gate_actions": int(step == 1),
                        "passing_candidate_gate_actions": score if step == 1 else 0,
                        "lambda_applied_visits": score if step == 1 else 0,
                        "beta_applied_visits": 0,
                    } for step in range(1, 6)}
                    (directory / "updates.jsonl").write_text(
                        "".join(json.dumps({
                            "update": update, "gate_by_step": gate_steps,
                            "candidate_gate_count": 1, "passed_gate_count": score,
                            "lambda_applied_visits": score,
                            "beta_applied_visits": 0,
                        }) + "\n"
                                for update in (1, 2, 3)), encoding="utf-8")
                    (directory / "validations.jsonl").write_text(
                        "".join(json.dumps({"update": update, "answer_f1": f1}) + "\n"
                                for update, f1 in ((0, 0.25), (3, 0.5))),
                        encoding="utf-8")
                    (directory / "best_selection.json").write_text(json.dumps({
                        "update": 3, "validation_answer_f1": 0.5,
                        "rule": config["checkpoint_selection"],
                        "audit_checkpoint": "checkpoints/state_00003.pt",
                    }), encoding="utf-8")
                    rows = [{
                        "question_id": question_id, "answer_f1_exact": str(score),
                        "answer_f1": float(score), "answer_em": score,
                        "read_count": 0, "read_order": [],
                        "gold_support_document_coverage": 0.0,
                        "termination": "early_submit",
                        "decisions": [{"completion_ids": [1, 0]}],
                        "answer_generation": {"completion_ids": [2, 0]},
                        "model_calls": 2, "generated_tokens": 4,
                    } for question_id in ("d0", "d1")]
                    with gzip.open(directory / "test.jsonl.gz", "wt",
                                   encoding="utf-8") as stream:
                        for row in rows:
                            stream.write(json.dumps(row) + "\n")
                    (directory / "test_metrics.json").write_text(json.dumps({
                        "question_count": 2, "answer_f1": float(score),
                        "answer_em": float(score), "model_calls": 4,
                        "generated_tokens": 8, "invalid_trajectories": 0,
                        "mean_read_count": 0.0,
                        "gold_support_document_coverage_mean": 0.0,
                        "full_gold_support_document_coverage_fraction": 0.0,
                        "selected_update": 3,
                        "selected_validation_answer_f1": 0.5,
                        "selected_checkpoint": "checkpoints/state_00003.pt",
                    }), encoding="utf-8")
            result = summarize(paths)
            self.assertEqual(len(result["per_seed"]), 3)
            self.assertEqual(result["mean_answer_f1_difference"], 1)
            self.assertEqual(result["mean_answer_em_difference"], 1)
            self.assertEqual(result["per_seed"][0]["two_step_training_gate_by_step"]
                             ["1"]["passing_candidate_gate_actions"], 3)
            self.assertEqual(result["per_seed"][0]["two_step_training_gate_by_step"]
                             ["1"]["candidate_gate_pass_rate"], 1)
            self.assertEqual(result["pooled_training_gate_by_step"]["two_step"]
                             ["1"]["passing_candidate_gate_actions"], 9)
            self.assertEqual(result["pooled_training_gate_by_step"]["baseline"]
                             ["1"]["candidate_gate_pass_rate"], 0)
            with self.assertRaisesRegex(ValueError, "Missing planned runs"):
                summarize(paths[:-1])

    def test_answer_f1_matches_hotpot_token_overlap_and_yes_no_rule(self) -> None:
        self.assertEqual(answer_em("The Paris!", "paris"), 1)
        self.assertEqual(answer_f1("red blue", "blue green"), Fraction(1, 2))
        self.assertEqual(answer_f1("yes", "no"), 0)
        self.assertEqual(answer_f1("yes indeed", "yes"), 0)
        self.assertEqual(answer_f1("yes", "yes"), 1)

    def test_fractional_f1_rewards_enter_existing_two_step_gate(self) -> None:
        root = ("question:1", "UNREAD", "1:Article 1")
        child = ("question:1", "read:1", "UNREAD")
        rewards = [Fraction(0), Fraction(1, 4), Fraction(1, 2),
                   Fraction(3, 4), Fraction(1)]
        visits = []
        for index, reward in enumerate(rewards, start=1):
            trajectory = f"t{index}"
            visits.append(Visit(f"{trajectory}:0", trajectory, 0, root,
                                "read:1", child, reward))
            visits.append(Visit(f"{trajectory}:1", trajectory, 1, child,
                                f"read:{index + 1}", None, reward))
        visits.append(Visit("other:0", "other", 0, root, "read:2", None, 0))
        visits.append(Visit("submit:0", "submit", 0, root, "submit", None, 1))
        result = compute_advantages(visits, omega=0, lambda_bonus=0.5,
                                    beta=1.5, seed=7)
        self.assertEqual(result.states[child].observed_legal_actions, 5)
        self.assertEqual(result.states[root].q["read:1"], Fraction(1, 2))
        self.assertTrue(result.gates[(root, "read:1")].passed)
        self.assertEqual(len(result.gates[(root, "read:1")].selected_second_actions), 3)
        by_step = _gate_by_step(result, arm="two_step")
        self.assertEqual(by_step["1"]["decision_visits"], 7)
        self.assertEqual(by_step["2"]["decision_visits"], 5)
        self.assertGreaterEqual(by_step["1"]["passing_candidate_gate_actions"], 1)
        self.assertEqual(by_step["2"]["nonterminal_candidate_gate_actions"], 0)
        self.assertEqual(
            by_step["1"]["candidate_gate_pass_rate"],
            by_step["1"]["passing_candidate_gate_actions"] /
            by_step["1"]["candidate_gate_actions"])
        baseline = _gate_by_step(result, arm="baseline")
        self.assertEqual(baseline["1"]["lambda_applied_visits"], 0)
        self.assertEqual(baseline["1"]["candidate_gate_pass_rate"],
                         by_step["1"]["candidate_gate_pass_rate"])
        one_decision = SimpleNamespace(record=lambda: {"decisions": [{
            "visit_id": "t1:0",
            "transition": {"before": root, "action_key": "read:1"},
        }]})
        joined = _trajectory_record_with_gate(one_decision, result,
                                              arm="two_step")["decisions"][0]
        self.assertEqual(joined["selection_step"], 1)
        self.assertTrue(joined["gate"]["passed"])
        self.assertEqual(joined["applied_advantage"],
                         result.advantages["t1:0"].final)


if __name__ == "__main__":
    unittest.main()
