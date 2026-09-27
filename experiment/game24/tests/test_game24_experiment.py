from fractions import Fraction
import unittest

from game24_experiment.env import Action, Game24, INVALID, legal_actions
from game24_experiment.method import Visit, _combine, compute_advantages
from game24_experiment.data import Puzzle, split_puzzles, load_puzzles, dataset_provenance, OFFICIAL_SHA256
from game24_experiment.config import ExperimentConfig
from pathlib import Path
import json
import gzip
import hashlib
import random
import torch
from game24_experiment.loss import decision_objective
from game24_experiment.modeling import Decision, PolicyRunner, score_logprobs, state_prompt_ids
from game24_experiment.train import _update, _stats_record, _source_hashes, main as train_main
from game24_experiment.checkpoints import save_training_state, file_sha256
from types import SimpleNamespace
import tempfile
from unittest.mock import patch
from game24_experiment.train import prepare
from game24_experiment.summarize import summarize


ROOT = Path(__file__).resolve().parents[1]


def completed_run_fixtures(root):
    config = json.loads((ROOT / "game24_experiment/experiment.json").read_text(encoding="utf-8"))
    config["model_revision"] = "a" * 40
    config["puzzles_per_update"] = 1136  # one update keeps the audit fixture small
    config["trajectories_per_puzzle"] = 1
    puzzles, digest = load_puzzles(ROOT / "data/24.csv")
    plan = {"schema_version": 2, "config": config,
            "tokenizer_revision": config["model_revision"],
            "special_tokens": {"eos_token_ids": [0, 1], "pad_token_id": 2},
            "dataset_provenance": dataset_provenance(),
            "split": split_puzzles(puzzles, digest, seed=config["split_seed"])}
    plan_bytes = (json.dumps(plan, indent=2) + "\n").encode()

    def evaluation(path, indices, successes):
        calls = tokens = 0
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for offset, index in enumerate(indices):
                reward = int(offset < successes)
                model_calls = 3 if reward else 1
                decisions = [{"completion_ids": [5, 1]} for _ in range(model_calls)]
                row = {"puzzle_index": index, "reward": reward,
                       "termination": "success" if reward else "invalid:format",
                       "steps": model_calls, "decisions": decisions,
                       "model_calls": model_calls, "generated_tokens": 2 * model_calls}
                stream.write(json.dumps(row) + "\n")
                calls += model_calls
                tokens += 2 * model_calls
        return {"successes": successes, "puzzles": len(indices),
                "accuracy": successes / len(indices),
                "invalid_trajectories": len(indices) - successes,
                "model_calls": calls, "generated_tokens": tokens}

    paths = []
    for arm in config["arms"]:
        for seed in config["training_seeds"]:
            path = root / f"{arm}-{seed}"
            path.mkdir()
            (path / "plan.json").write_bytes(plan_bytes)
            plan_hash = hashlib.sha256(plan_bytes).hexdigest()
            (path / "manifest.json").write_text(json.dumps({
                "schema_version": 2, "status": "complete", "arm": arm, "seed": seed,
                "plan": plan, "plan_sha256": plan_hash,
                "versions": {"fixture": "1"}, "completed_updates": 1,
                "selected_update": 0,
            }), encoding="utf-8")
            order = list(plan["split"]["train"])
            random.Random(seed).shuffle(order)
            (path / "train_order.json").write_text(json.dumps({"indices": order, "seed": seed}))
            checkpoint_records = []
            for update in (0, 1):
                snapshot = path / "checkpoints" / f"state_{update:04d}.pt"
                snapshot.parent.mkdir(exist_ok=True)
                snapshot.write_bytes(f"fixture checkpoint {update}".encode())
                checkpoint_records.append({
                    "file": f"checkpoints/state_{update:04d}.pt", "update": update,
                    "arm": arm, "seed": seed, "plan_sha256": plan_hash,
                    "bytes": snapshot.stat().st_size, "sha256": file_sha256(snapshot),
                })
            (path / "checkpoints.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in checkpoint_records),
            )
            for name in ("rollouts_0001.jsonl.gz", "tokens_0001.jsonl.gz", "stats_0001.json"):
                (path / name).write_bytes(b"fixture artifact")
            artifacts = {
                name: {"bytes": (path / name).stat().st_size,
                       "sha256": file_sha256(path / name)}
                for name in ("rollouts_0001.jsonl.gz", "tokens_0001.jsonl.gz",
                             "stats_0001.json")
            }
            update_record = {
                "update": 1, "epoch": 0, "puzzle_indices": order,
                "old_policy_checkpoint": checkpoint_records[0]["file"],
                "new_policy_checkpoint": checkpoint_records[1]["file"],
                "new_checkpoint_sha256": checkpoint_records[1]["sha256"],
                "trajectory_count": 1136, "decision_count": 1136,
                "generated_tokens": 2272, "invalid_trajectories": 1126,
                "candidate_gate_count": 100, "passed_gate_count": 20,
                "lambda_eligible_visit_count": 50, "beta_eligible_visit_count": 30,
                "lambda_visit_count": 50 if arm == "two_step" else 0,
                "beta_selected_visit_count": 30 if arm == "two_step" else 0,
                "overlap_visit_count": 5 if arm == "two_step" else 0,
                "update_artifacts": artifacts,
            }
            (path / "updates.jsonl").write_text(json.dumps(update_record) + "\n")
            validation_rows = []
            for update in (0, 1):
                metrics = evaluation(path / f"validation_{update:04d}.jsonl.gz",
                                     plan["split"]["validation"], 63)
                validation_rows.append({"update": update, **metrics})
            (path / "validations.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in validation_rows),
            )
            (path / "best_selection.json").write_text(json.dumps({
                "update": 0, "validation_accuracy": 0.5,
                "audit_checkpoint": checkpoint_records[0]["file"],
            }))
            for adapter in ("best_adapter", "final_adapter"):
                folder = path / adapter
                folder.mkdir()
                (folder / "adapter_config.json").write_text("{}")
                (folder / "adapter_model.safetensors").write_bytes(b"fixture")
            test_metrics = evaluation(path / "test.jsonl.gz", plan["split"]["test"],
                                      20 if arm == "baseline" else 40)
            test_metrics.update(selected_update=0, selected_validation_accuracy=0.5,
                                selected_checkpoint=checkpoint_records[0]["file"])
            (path / "test_metrics.json").write_text(json.dumps(test_metrics))
            paths.append(path)
    return paths


class EnvironmentTests(unittest.TestCase):
    def test_three_separate_calls_reach_24(self):
        game = Game24([4, 5, 6, 10])
        self.assertEqual(game.observation, "[1:4, 2:5, 3:6, 4:10]")
        first = game.step("4 1 -")
        self.assertEqual(first.after, ("5", "6", "6"))
        self.assertFalse(first.terminal)
        second = game.step("1 2 *")
        self.assertEqual(second.after, ("6", "30"))
        last = game.step("2 1 -")
        self.assertEqual(last.reward, 1)
        self.assertTrue(last.terminal)
        self.assertEqual(game.termination, "success")

    def test_duplicate_slots_are_distinct_and_commutation_is_not(self):
        self.assertNotEqual(Action(1, 2, "*").key, Action(1, 3, "*").key)
        self.assertEqual(Action(1, 2, "*").key, Action(2, 1, "*").key)
        self.assertNotEqual(Action(1, 2, "-").key, Action(2, 1, "-").key)
        self.assertEqual(len(legal_actions([Fraction(5), Fraction(6), Fraction(6)])), 18)

    def test_exact_fraction_and_invalid_terminal_reward(self):
        game = Game24([1, 2, 3, 4])
        self.assertIn("1/2", game.step("1 2 /").after)
        bad = game.step("1 1 +")
        self.assertEqual(bad.action_key, INVALID)
        self.assertEqual(bad.error, "same_slot_twice")
        self.assertEqual(bad.reward, 0)
        self.assertTrue(bad.terminal)

    def test_zero_division_is_invalid(self):
        game = Game24([1, 2, 2, 3])
        game.step("2 3 -")
        zero_slot = game.state_key.index("0") + 1
        other_slot = 1 if zero_slot != 1 else 2
        transition = game.step(f"{other_slot} {zero_slot} /")
        self.assertEqual(transition.error, "division_by_zero")
        self.assertEqual(transition.reward, 0)


class AdvantageTests(unittest.TestCase):
    def test_five_legal_actions_gate_and_position_level_lambda(self):
        root = ("4", "5", "6", "10")
        child = ("5", "6", "6")
        rows = []
        rewards = [0, 0, 1, 1, 1]
        for i, reward in enumerate(rewards):
            trajectory = f"a{i}"
            rows.append(Visit(f"{trajectory}-0", trajectory, 0, root, "-:4:1", child, reward))
            rows.append(Visit(f"{trajectory}-1", trajectory, 1, child, f"a{i}", (str(i),), reward))
        for i in range(3):
            rows.append(Visit(f"b{i}-0", f"b{i}", 0, root, "+:1:2", None, 0))
            rows.append(Visit(f"c{i}-0", f"c{i}", 0, root, "*:1:2", None, 1))
        rows.append(Visit("invalid-0", "invalid", 0, child, INVALID, None, 0))
        batch = compute_advantages(rows, omega=1.0, lambda_bonus=0.5, beta=1.5, seed=7)
        gate = batch.gates[(root, "-:4:1")]
        self.assertTrue(gate.passed)
        self.assertEqual(gate.observed_legal_actions, 5)
        self.assertEqual(gate.selection_count, 3)
        self.assertNotIn(INVALID, gate.selected_second_actions)
        selected_first = [i for i in range(5) if batch.advantages[f"a{i}-0"].B]
        self.assertEqual(len(selected_first), 3)
        for i in range(5):
            first = batch.advantages[f"a{i}-0"]
            self.assertAlmostEqual(first.final - first.base, 0.5 if i in selected_first else 0)
            second = batch.advantages[f"a{i}-1"]
            self.assertTrue(second.P)  # action-level gate applies even when first visit was not selected
            expected = 1.5 * second.base if second.selected_second else second.base
            self.assertAlmostEqual(second.final, expected)
        trajectories = [SimpleNamespace(decisions=[SimpleNamespace(transition=SimpleNamespace(
            before=row.state, action_key=row.action, after=row.child_state,
        )) for row in rows])]
        two = _stats_record(trajectories, batch, arm="two_step")
        base = _stats_record(trajectories, batch, arm="baseline")
        root_two = next(row for row in two["states"] if row["state"] == root)["gate_summary"]
        root_base = next(row for row in base["states"] if row["state"] == root)["gate_summary"]
        child_two = next(row for row in two["states"] if row["state"] == child)["gate_summary"]
        self.assertEqual(root_two["visit_count"], 11)
        self.assertEqual(root_two["passed_action_count"], 1)
        self.assertEqual(root_two["lambda_applied_visit_count"], 3)
        self.assertEqual(child_two["beta_applied_visit_count"], 3)
        self.assertEqual(root_base["lambda_eligible_visit_count"], 3)
        self.assertEqual(root_base["lambda_applied_visit_count"], 0)
        self.assertEqual(base["gate_usage"], "counterfactual_diagnostic_only")
        self.assertTrue(all(row["applied_advantage"] == row["base"] for row in base["advantages"]))

    def test_confirmed_overlap_is_additive(self):
        final, first, second = _combine(0.8, True, True, True, 0.5, 1.5)
        self.assertAlmostEqual(final, 0.8 + 0.5 + 1.5 * 0.8)
        self.assertAlmostEqual(first, 0.5)
        self.assertAlmostEqual(second, 1.5 * 0.8)

    def test_zero_variance_has_zero_advantage_and_no_gate(self):
        state = ("1", "2", "3", "4")
        rows = [
            Visit("a", "ta", 0, state, "+:1:2", None, 0),
            Visit("b", "tb", 0, state, "-:1:2", None, 0),
        ]
        batch = compute_advantages(rows, omega=1, lambda_bonus=0.5, beta=1.5, seed=1)
        self.assertEqual(batch.selected_first_visits, set())
        self.assertEqual(batch.advantages["a"].final, 0)
        self.assertEqual(batch.advantages["b"].final, 0)


class ConfigurationTests(unittest.TestCase):
    def test_source_hashes_use_portable_paths(self):
        hashes = _source_hashes()
        self.assertIn("game24_experiment/train.py", hashes)
        self.assertTrue(all("\\" not in path for path in hashes))

    def test_split_is_fixed_disjoint_and_official_test_is_untouched(self):
        puzzles = [Puzzle(index, (1, 2, 3, index + 4)) for index in range(1362)]
        first = split_puzzles(puzzles, "fixture", seed=20260926)
        second = split_puzzles(puzzles, "fixture", seed=20260926)
        self.assertEqual(first, second)
        self.assertEqual((len(first["train"]), len(first["validation"]), len(first["test"])),
                         (1136, 126, 100))
        self.assertEqual(first["test"], list(range(900, 1000)))
        self.assertEqual(len(set(first["train"] + first["validation"] + first["test"])), 1362)

    def test_first_run_config_is_complete(self):
        path = Path(__file__).resolve().parents[1] / "game24_experiment" / "experiment.json"
        config = ExperimentConfig.from_json(path)
        self.assertEqual(config.trajectories_per_puzzle, 256)
        self.assertEqual(config.kl_coefficient, 0.01)

    def test_prepare_locks_dataset_split_and_model_revision(self):
        from transformers import GenerationConfig

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = ROOT / "data/24.csv"
            source = Path(__file__).resolve().parents[1] / "game24_experiment" / "experiment.json"
            config = json.loads(source.read_text(encoding="utf-8"))
            config["dataset_csv"] = str(csv_path)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with patch("huggingface_hub.HfApi.model_info", return_value=SimpleNamespace(sha="a" * 40)), \
                 patch("transformers.GenerationConfig.from_pretrained",
                       return_value=GenerationConfig(eos_token_id=[0, 1], pad_token_id=2)) as generation:
                prepare(config_path, root / "plan")
            generation.assert_called_once_with(config["model_id"], revision="a" * 40)
            plan = json.loads((root / "plan" / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["config"]["model_revision"], "a" * 40)
            self.assertEqual(len(plan["split"]["validation"]), 126)
            self.assertEqual(plan["split"]["test"], list(range(900, 1000)))
            self.assertEqual(plan["special_tokens"], {"eos_token_ids": [0, 1], "pad_token_id": 2})
            self.assertEqual(plan["dataset_provenance"]["sha256"], OFFICIAL_SHA256)
            self.assertEqual(plan["tokenizer_revision"], "a" * 40)

    def test_official_data_rejects_reordered_rows_with_same_count_and_index_900(self):
        raw = (ROOT / "data/24.csv").read_bytes()
        lines = raw.splitlines(keepends=True)
        lines[1], lines[2] = lines[2], lines[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "24.csv"
            path.write_bytes(b"".join(lines))
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_puzzles(path)

    def test_summary_pairs_arms_on_same_puzzles_and_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = completed_run_fixtures(Path(directory))
            report = summarize(paths, bootstrap_replicates=200)
            self.assertAlmostEqual(report["mean_baseline"], 0.2)
            self.assertAlmostEqual(report["mean_two_step"], 0.4)
            self.assertAlmostEqual(report["paired_difference"], 0.2)
            self.assertEqual(report["run_count"], 6)
            self.assertEqual(report["by_arm_budget_and_gates"]["baseline"]
                             ["training_generation_decisions"], 3 * 1136)
            self.assertEqual(report["by_arm_budget_and_gates"]["two_step"]
                             ["lambda_applied_visits"], 3 * 50)
            self.assertEqual(report["by_arm_budget_and_gates"]["baseline"]
                             ["lambda_applied_visits"], 0)
            missing_pair = [path for path in paths if not path.name.endswith("20260928")]
            with self.assertRaisesRegex(ValueError, "Incomplete planned experiment"):
                summarize(missing_pair, bootstrap_replicates=200)

    def test_summary_rejects_changed_plan_and_incomplete_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = completed_run_fixtures(Path(directory))
            plan_path = paths[0] / "plan.json"
            raw = plan_path.read_bytes()
            plan_path.write_bytes(raw + b" ")
            with self.assertRaisesRegex(ValueError, "altered locked plan"):
                summarize(paths)
            plan_path.write_bytes(raw)
            manifest_path = paths[0] / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["completed_updates"] = 0
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "planned updates"):
                summarize(paths)

    def test_summary_rejects_missing_records_bad_snapshot_and_wrong_best_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = completed_run_fixtures(Path(directory))
            first = paths[0]
            update_log = first / "updates.jsonl"
            raw = update_log.read_bytes()
            update_log.unlink()
            with self.assertRaisesRegex(ValueError, "Missing run record"):
                summarize(paths, bootstrap_replicates=200)
            update_log.write_bytes(raw)
            stats = first / "stats_0001.json"
            original_stats = stats.read_bytes()
            stats.write_bytes(b"altered statistics")
            with self.assertRaisesRegex(ValueError, "altered update artifact"):
                summarize(paths, bootstrap_replicates=200)
            stats.write_bytes(original_stats)
            snapshot = first / "checkpoints/state_0001.pt"
            raw = snapshot.read_bytes()
            snapshot.write_bytes(b"altered snapshot")
            with self.assertRaisesRegex(ValueError, "altered audit checkpoint"):
                summarize(paths, bootstrap_replicates=200)
            snapshot.write_bytes(raw)
            selection = first / "best_selection.json"
            selected = json.loads(selection.read_text())
            selected["update"] = 1
            selection.write_text(json.dumps(selected))
            with self.assertRaisesRegex(ValueError, "Best checkpoint"):
                summarize(paths, bootstrap_replicates=200)

    def test_refused_output_reuse_does_not_corrupt_existing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = path / "manifest.json"
            original = '{"status": "complete"}\n'
            manifest.write_text(original)
            with patch("game24_experiment.train.run", side_effect=FileExistsError("not empty")):
                with self.assertRaises(FileExistsError):
                    train_main(["run", "--plan", "unused.json", "--output", str(path),
                                "--arm", "baseline", "--seed", "20260926"])
            self.assertEqual(manifest.read_text(), original)


class TokenTests(unittest.TestCase):
    def test_clipped_objective_is_token_mean_with_differentiable_kl(self):
        current = torch.tensor([-0.1, -1.2], requires_grad=True)
        old = torch.tensor([-0.2, -1.0])
        reference = torch.tensor([-0.3, -0.8])
        value, details = decision_objective(current, old, reference, 0.7,
                                            epsilon=0.2, kappa=0.01)
        ratio = torch.exp(current - old)
        policy = torch.minimum(ratio * 0.7, ratio.clamp(0.8, 1.2) * 0.7)
        log_ref_over_current = reference - current
        kl = torch.exp(log_ref_over_current) - log_ref_over_current - 1
        self.assertTrue(torch.allclose(value, (policy - 0.01 * kl).mean()))
        value.backward()
        self.assertIsNotNone(current.grad)
        self.assertEqual(len(details["ratio"]), 2)

    def test_clip_active_mask_depends_on_advantage_sign(self):
        old = torch.zeros(2)
        ref = torch.zeros(2)
        current = torch.log(torch.tensor([0.7, 1.3]))
        _, positive = decision_objective(current, old, ref, 1.0,
                                         epsilon=0.2, kappa=0.01)
        _, negative = decision_objective(current, old, ref, -1.0,
                                         epsilon=0.2, kappa=0.01)
        self.assertEqual(positive["clipped"].tolist(), [False, True])
        self.assertEqual(negative["clipped"].tolist(), [True, False])

    def test_scoring_uses_only_generated_token_targets(self):
        class FakeModel:
            def __call__(self, input_ids, attention_mask, use_cache):
                logits = torch.zeros((*input_ids.shape, 10), dtype=torch.float32)
                logits.scatter_(-1, ((input_ids + 1) % 10).unsqueeze(-1), 3.0)
                return type("Output", (), {"logits": logits})()

        transition = Game24([1, 2, 3, 4]).step("1 2 +")
        rows = [
            Decision("a", "", [1, 2], [3, 4], ["3", "4"], True, transition),
            Decision("b", "", [1, 2, 3], [4], ["4"], True, transition),
        ]
        scores = score_logprobs(FakeModel(), rows, pad_id=0, device=torch.device("cpu"))
        self.assertEqual([len(row) for row in scores], [2, 1])
        self.assertTrue(all(score.item() > -1 for row in scores for score in row))

    def test_prompt_contains_current_state_and_no_history(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                                    return_dict):
                self.return_dict = return_dict
                self.messages = messages
                return [1, 2, 3]

            def decode(self, ids, skip_special_tokens):
                return "prompt"

        tokenizer = FakeTokenizer()
        game = Game24([4, 5, 6, 10])
        game.step("4 1 -")
        _, ids = state_prompt_ids(tokenizer, game)
        self.assertEqual(ids, [1, 2, 3])
        self.assertFalse(tokenizer.return_dict)
        self.assertIn("[1:5, 2:6, 3:6]", tokenizer.messages[-1]["content"])
        self.assertNotIn("10", tokenizer.messages[-1]["content"])

    def test_runner_makes_three_fresh_model_calls(self):
        class CharacterTokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                                    return_dict):
                assert return_dict is False
                content = messages[-1]["content"]
                return [ord(char) + 1 for char in content]

            def decode(self, ids, skip_special_tokens):
                return "".join(chr(token - 1) for token in ids if token != 0)

            def convert_ids_to_tokens(self, ids):
                return ["EOS" if token == 0 else chr(token - 1) for token in ids]

        class ScriptedModel:
            def __init__(self):
                self.prompts = []
                self.generation_config = SimpleNamespace(eos_token_id=[0, 1], pad_token_id=2)

            def eval(self):
                return self

            def generate(self, **kwargs):
                input_ids = kwargs["input_ids"]
                mask = kwargs["attention_mask"]
                outputs = []
                for row, row_mask in zip(input_ids, mask):
                    prompt = "".join(chr(token - 1) for token, keep in zip(row.tolist(), row_mask.tolist()) if keep)
                    self.prompts.append(prompt)
                    state = prompt.split("Current numbers: ")[-1]
                    command = {
                        "[1:4, 2:5, 3:6, 4:10]": "4 1 -",
                        "[1:5, 2:6, 3:6]": "1 2 *",
                        "[1:6, 2:30]": "2 1 -",
                    }[state]
                    outputs.append([ord(char) + 1 for char in command] + [len(self.prompts) % 2])
                width = max(len(row) for row in outputs)
                tail = torch.tensor([row + [2] * (width - len(row)) for row in outputs])
                return torch.cat([input_ids, tail], dim=1)

        model = ScriptedModel()
        runner = PolicyRunner(model, CharacterTokenizer(), max_new_tokens=32,
                              generation_batch_size=2, device=torch.device("cpu"))
        trajectories = runner.rollout([(900, (4, 5, 6, 10))], repetitions=1,
                                      label="fixture", do_sample=False)
        self.assertEqual(trajectories[0].reward, 1)
        self.assertEqual(len(model.prompts), 3)
        self.assertTrue(model.prompts[1].endswith("[1:5, 2:6, 3:6]"))
        self.assertNotIn("[1:4, 2:5, 3:6, 4:10]", model.prompts[1])
        self.assertEqual(trajectories[0].decisions[0].record()["termination_token_id"], 1)
        self.assertEqual(trajectories[0].decisions[1].record()["termination_token_id"], 0)

    def test_generation_trims_both_eos_ids_and_excludes_padding_from_targets(self):
        class Model:
            generation_config = SimpleNamespace(eos_token_id=[0, 1], pad_token_id=2)

            def generate(self, **kwargs):
                self.arguments = kwargs
                tail = torch.tensor([[5, 1, 2, 2], [6, 7, 8, 0]])
                return torch.cat([kwargs["input_ids"], tail], dim=1)

        model = Model()
        runner = PolicyRunner(model, SimpleNamespace(eos_token_id=0), max_new_tokens=4,
                              generation_batch_size=2, device=torch.device("cpu"))
        generated = runner._generate([[10, 11], [12]], do_sample=True)
        self.assertEqual(generated, [[5, 1], [6, 7, 8, 0]])
        self.assertEqual(model.arguments["eos_token_id"], [0, 1])
        self.assertEqual(model.arguments["pad_token_id"], 2)
        self.assertEqual(model.arguments["input_ids"][1].tolist(), [2, 12])
        self.assertEqual(model.arguments["attention_mask"][1].tolist(), [0, 1])

    def test_training_update_changes_parameters_and_writes_token_audit(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(10, 6)
                self.output = torch.nn.Linear(6, 10)

            def forward(self, input_ids, attention_mask, use_cache):
                return SimpleNamespace(logits=self.output(self.embedding(input_ids)))

        model = TinyModel()
        transition = Game24([1, 2, 3, 4]).step("1 2 +")
        decision = Decision("visit", "prompt", [1, 2], [3, 4], ["3", "4"],
                            True, transition)
        with torch.no_grad():
            old = score_logprobs(model, [decision], pad_id=0,
                                  device=torch.device("cpu"))[0].tolist()
        decision.old_logp = old
        decision.reference_logp = old
        trajectory = SimpleNamespace(trajectory_id="trajectory", decisions=[decision])
        advantage = SimpleNamespace(advantages={"visit": SimpleNamespace(base=1.0, final=1.5)})
        config = ExperimentConfig.from_json(
            Path(__file__).resolve().parents[1] / "game24_experiment" / "experiment.json"
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
        before = model.output.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.jsonl.gz"
            metrics = _update(model, optimizer, [trajectory], advantage, arm="baseline",
                              config=config, pad_id=0, device=torch.device("cpu"),
                              microbatch_size=1, token_log_path=path)
            self.assertTrue(path.is_file())
            self.assertEqual(metrics["generated_tokens"], 2)
        self.assertFalse(torch.allclose(before, model.output.weight))


if __name__ == "__main__":
    unittest.main()
