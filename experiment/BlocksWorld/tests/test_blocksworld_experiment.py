"""Verify six-block data rules and shared GRPO method integration."""

import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from blocksworld_experiment.config import ExperimentConfig
from blocksworld_experiment.data import Instance, load_instances, split_instances
from blocksworld_experiment.domain import Atom, Problem, State
from blocksworld_experiment.env import Blocksworld
from blocksworld_experiment.generate import PLANBENCH_REPOSITORY, PLANBENCH_REVISION
from blocksworld_experiment.modeling import PolicyRunner
from blocksworld_experiment.summarize import _evaluation, summarize
from blocksworld_experiment.train import _stats_record, _visits, prepare, run
from game24_experiment.method import compute_advantages


def _pddl(goal_support: str) -> str:
    return f"""(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f)
(:init (handempty) (ontable b) (on a b) (clear a)
       (ontable c) (clear c) (ontable d) (clear d)
       (ontable e) (clear e) (ontable f) (clear f))
(:goal (and (on a {goal_support})))
)"""


class ExperimentTests(unittest.TestCase):
    def test_config_is_complete_and_separate_from_game24(self) -> None:
        path = Path("blocksworld_experiment/experiment.json")
        config = ExperimentConfig.from_json(path)
        self.assertEqual(config.dataset_size, 500)
        self.assertEqual((config.train_size, config.validation_size, config.test_size),
                         (400, 50, 50))
        self.assertEqual(config.max_steps, 16)

    def test_dataset_hashes_and_unfiltered_deterministic_split(self) -> None:
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            entries = []
            for index, target in enumerate(("c", "d"), start=1):
                name = f"instance-{index:04d}.pddl"
                raw = _pddl(target).encode()
                (directory / name).write_bytes(raw)
                entries.append({"file": name, "sha256": hashlib.sha256(raw).hexdigest()})
            manifest = {
                "schema_version": 1,
                "source_repository": PLANBENCH_REPOSITORY,
                "source_revision": PLANBENCH_REVISION,
                "block_count": 6,
                "instance_count": 2,
                "length_filter": None,
                "instances": entries,
            }
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            instances, _ = load_instances(directory, expected_count=2)
            self.assertEqual([instance.shortest_length for instance in instances], [2, 2])
            split = split_instances(instances, seed=7, train_size=1,
                                    validation_size=1, test_size=0)
            self.assertEqual(sorted(split["train"]+split["validation"]), [0, 1])
            self.assertEqual(split["over_16_steps"],
                             {"train": 0, "validation": 0, "test": 0})
            (directory / entries[0]["file"]).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                load_instances(directory, expected_count=2)

    def test_available_actions_are_not_observed_actions_in_gate(self) -> None:
        start = State((("b", "a"), ("c",), ("d",), ("e",), ("f",)))
        problem = Problem(start.blocks, start, (Atom("on", ("a", "c")),))

        def trajectory(identifier: str, moves: list[str]) -> SimpleNamespace:
            game = Blocksworld(problem)
            decisions = []
            for step, move in enumerate(moves):
                transition = game.step(move)
                decisions.append(SimpleNamespace(
                    visit_id=f"{identifier}:{step}", transition=transition,
                ))
            return SimpleNamespace(trajectory_id=identifier, game=game,
                                   decisions=decisions, reward=game.reward)

        good = trajectory("good", ["unstack a b", "stack a c"])
        bad = trajectory("bad", ["pick-up b"])
        rows = _visits([good, bad])
        batch = compute_advantages(rows, omega=1.0, lambda_bonus=0.5,
                                   beta=1.5, seed=3)
        stats = _stats_record([good, bad], batch, arm="baseline")
        root = next(row for row in stats["states"] if row["state"] == rows[0].state)
        self.assertEqual(root["distinct_legal_actions_available"], 5)
        self.assertEqual(root["observed_legal_actions"], 1)
        self.assertEqual(stats["gate_usage"], "counterfactual_diagnostic_only")
        self.assertTrue(all(row["applied_advantage"] == row["base"]
                            for row in stats["advantages"]))

    def test_runner_refreshes_symbolic_state_and_goal_each_action(self) -> None:
        class CharacterTokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                                    return_dict):
                assert return_dict is False
                return [ord(char)+1 for char in messages[-1]["content"]]

            def decode(self, ids, skip_special_tokens):
                return "".join(chr(token-1) for token in ids if token != 0)

            def convert_ids_to_tokens(self, ids):
                return ["EOS" if token == 0 else chr(token-1) for token in ids]

        class ScriptedModel:
            def __init__(self):
                self.prompts = []
                self.generation_config = SimpleNamespace(eos_token_id=0, pad_token_id=2)

            def eval(self):
                return self

            def generate(self, **kwargs):
                input_ids = kwargs["input_ids"]
                mask = kwargs["attention_mask"]
                outputs = []
                for row, row_mask in zip(input_ids, mask):
                    prompt = "".join(chr(token-1) for token, keep in
                                     zip(row.tolist(), row_mask.tolist()) if keep)
                    self.prompts.append(prompt)
                    command = "stack a c" if "holding(a)" in prompt else "unstack a b"
                    outputs.append([ord(char)+1 for char in command] + [0])
                width = max(map(len, outputs))
                tail = torch.tensor([row + [2]*(width-len(row)) for row in outputs])
                return torch.cat([input_ids, tail], dim=1)

        start = State((("b", "a"), ("c",), ("d",), ("e",), ("f",)))
        problem = Problem(start.blocks, start, (Atom("on", ("a", "c")),))
        instance = Instance(0, "fixture.pddl", "x", problem, 2)
        model = ScriptedModel()
        runner = PolicyRunner(model, CharacterTokenizer(), max_new_tokens=32,
                              generation_batch_size=2, device=torch.device("cpu"))
        trajectories = runner.rollout([(0, instance)], repetitions=1,
                                      label="fixture", do_sample=False)
        self.assertEqual(trajectories[0].reward, 1)
        self.assertEqual(len(model.prompts), 2)
        self.assertIn("Actions remaining: 16", model.prompts[0])
        self.assertIn("Actions remaining: 15", model.prompts[1])
        self.assertIn("Goal: on(a,c)", model.prompts[1])
        self.assertEqual(trajectories[0].record()["model_calls"], 2)

    def test_prepare_locks_three_synthetic_instances_and_run_stays_off_cpu(self) -> None:
        with TemporaryDirectory() as folder:
            base = Path(folder)
            dataset = base / "data"
            dataset.mkdir()
            entries = []
            for index, target in enumerate(("c", "d", "e"), start=1):
                name = f"instance-{index:04d}.pddl"
                raw = _pddl(target).encode()
                (dataset / name).write_bytes(raw)
                entries.append({"file": name, "sha256": hashlib.sha256(raw).hexdigest()})
            manifest = {
                "schema_version": 1,
                "source_repository": PLANBENCH_REPOSITORY,
                "source_revision": PLANBENCH_REVISION,
                "generator_command": "./blocksworld 4 6",
                "generator_files": {},
                "candidate_calls": 3,
                "rejected_trivial_initial_goal": 0,
                "rejected_duplicate_initial_goal": 0,
                "block_count": 6,
                "instance_count": 3,
                "length_filter": None,
                "instances": entries,
            }
            (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            settings = json.loads(Path("blocksworld_experiment/experiment.json").read_text())
            settings.update(dataset_dir=str(dataset), dataset_size=3, train_size=1,
                            validation_size=1, test_size=1,
                            training_seeds=[20260926], model_revision="a"*40)
            config_path = base / "config.json"
            config_path.write_text(json.dumps(settings), encoding="utf-8")
            plan_dir = base / "plan"
            with patch("transformers.GenerationConfig.from_pretrained",
                       return_value=SimpleNamespace(eos_token_id=[0, 1], pad_token_id=2)):
                prepare(config_path, plan_dir)
            plan = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["special_tokens"],
                             {"eos_token_ids": [0, 1], "pad_token_id": 2})
            self.assertEqual(plan["split"]["over_16_steps"],
                             {"train": 0, "validation": 0, "test": 0})
            with patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "requires a CUDA GPU"):
                    run(plan_dir / "plan.json", base / "no_gpu_run",
                        arm="baseline", seed=20260926)
            self.assertFalse((base / "no_gpu_run").exists())

    def test_evaluation_trace_and_paired_summary(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "test.jsonl.gz"
            record = {
                "problem_index": 0, "reward": 1, "termination": "success",
                "steps": 2, "legal_actions_executed": 2, "model_calls": 2,
                "generated_tokens": 4,
                "decisions": [{"completion_ids": [11, 0]},
                              {"completion_ids": [12, 0]}],
            }
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            outcomes, metrics = _evaluation(path, [0], max_steps=16)
            self.assertEqual(outcomes, {0: 1})
            self.assertEqual(metrics["accuracy"], 1)
            record["steps"] = 1
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            with self.assertRaises(ValueError):
                _evaluation(path, [0], max_steps=16)

        plan = {"config": {"arms": ["baseline", "two_step"],
                           "training_seeds": [7]},
                "split": {"test": [0, 1], "over_16_steps": {"test": 0}}}
        common = {"plan": plan, "versions": {"torch": "test"},
                  "plan_sha256": "locked"}
        fixtures = [
            ({**common, "arm": "baseline", "seed": 7}, {0: 0, 1: 1},
             {"training_generation_decisions": 12}),
            ({**common, "arm": "two_step", "seed": 7}, {0: 1, 1: 1},
             {"training_generation_decisions": 12}),
        ]
        with patch("blocksworld_experiment.summarize._load_run", side_effect=fixtures):
            summary = summarize([Path("base"), Path("two")], bootstrap_replicates=100)
        self.assertEqual(summary["mean_baseline"], 0.5)
        self.assertEqual(summary["mean_two_step"], 1.0)
        self.assertEqual(summary["paired_difference"], 0.5)
        with patch("blocksworld_experiment.summarize._load_run", return_value=fixtures[0]):
            with self.assertRaises(ValueError):
                summarize([Path("base")], bootstrap_replicates=100)


if __name__ == "__main__":
    unittest.main()
