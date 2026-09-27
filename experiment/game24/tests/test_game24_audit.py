from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import torch

from game24_experiment.checkpoints import file_sha256, save_training_state
from game24_experiment.config import ExperimentConfig
from game24_experiment.env import Game24, INVALID
from game24_experiment.method import Visit, compute_advantages
from game24_experiment.modeling import Decision, PolicyRunner, score_logprobs, state_prompt_ids
from game24_experiment.train import (
    _attach_old_and_reference_logprobs, _diagnostic_metrics, _stats_record, _update,
)


ROOT = Path(__file__).resolve().parents[1]


class AuditTests(unittest.TestCase):
    def test_real_transformers_chat_template_reaches_qwen_generation(self):
        """Exercise the installed tokenizer API and a real Qwen generate call."""
        from peft import LoraConfig, get_peft_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

        backend = Tokenizer(WordLevel(
            {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2}, unk_token="[UNK]",
        ))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]",
            eos_token="[EOS]", pad_token="[PAD]",
        )
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        )
        _, prompt_ids = state_prompt_ids(tokenizer, Game24([1, 2, 3, 4]))
        self.assertEqual(prompt_ids, [0])

        torch.manual_seed(29)
        tiny = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=16, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=128, eos_token_id=1, pad_token_id=2,
        ))
        model = get_peft_model(tiny, LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        ))
        model.enable_input_require_grads()
        runner = PolicyRunner(model, tokenizer, max_new_tokens=3,
                              generation_batch_size=2, device=torch.device("cpu"))
        trajectories = runner.rollout([(0, (1, 2, 3, 4))], repetitions=1,
                                      label="real-api", do_sample=False)
        self.assertEqual(len(trajectories), 1)
        self.assertGreaterEqual(len(trajectories[0].decisions), 1)
        self.assertTrue(all(type(token) is int for token in
                            trajectories[0].decisions[0].prompt_ids))

    def test_checkpoint_restores_weights_optimizer_and_random_state(self):
        torch.manual_seed(31)
        random.seed(31)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        def step(current, opt):
            opt.zero_grad(set_to_none=True)
            current(torch.tensor([[0.3, -0.2]])).square().sum().backward()
            opt.step()

        step(model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            record = save_training_state(path, model, optimizer, metadata={"update": 1})
            self.assertEqual(record["sha256"], file_sha256(path))
            payload = torch.load(path, weights_only=True)
            expected_random = (random.random(), torch.rand(3))
            step(model, optimizer)
            expected_weights = {name: parameter.detach().clone()
                                for name, parameter in model.named_parameters()}
            restored = torch.nn.Linear(2, 1)
            restored.load_state_dict(payload["trainable_parameters"])
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=999)
            restored_optimizer.load_state_dict(payload["optimizer"])
            random.setstate(payload["rng"]["python"])
            torch.set_rng_state(payload["rng"]["torch_cpu"])
            self.assertEqual(random.random(), expected_random[0])
            self.assertTrue(torch.equal(torch.rand(3), expected_random[1]))
            step(restored, restored_optimizer)
            for name, parameter in restored.named_parameters():
                self.assertTrue(torch.equal(parameter, expected_weights[name]))
            with self.assertRaises(FileExistsError):
                save_training_state(path, model, optimizer, metadata={"update": 2})

    def test_zero_success_diagnostic_does_not_invent_advantages(self):
        game = Game24([1, 2, 3, 4])
        transition = game.fail("bad", "format")
        rows = [Visit("v", "t", 0, transition.before, INVALID, None, 0)]
        batch = compute_advantages(rows, omega=1, lambda_bonus=0.5, beta=1.5, seed=7)
        trajectories = [SimpleNamespace(reward=0, game=game,
                                        decisions=[SimpleNamespace(transition=transition)])]
        metrics = _diagnostic_metrics(trajectories, batch, arm="two_step")
        self.assertIn("zero_success_batch", metrics["diagnostic_flags"])
        self.assertIn("zero_policy_advantages", metrics["diagnostic_flags"])
        summary = _stats_record(trajectories, batch, arm="two_step")["states"][0]["gate_summary"]
        self.assertEqual(summary["candidate_action_count"], 0)
        self.assertIsNone(summary["passed_candidate_action_fraction"])
        self.assertEqual(summary["lambda_applied_visit_count"], 0)
        self.assertEqual(batch.advantages["v"].final, 0)

    def test_valid_text_without_eos_still_ends_as_generation_limit(self):
        class Tokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                                    return_dict):
                assert return_dict is False
                return [5]

            def decode(self, ids, skip_special_tokens):
                return "1 2 +"

            def convert_ids_to_tokens(self, ids):
                return [str(token) for token in ids]

        class Model:
            generation_config = SimpleNamespace(eos_token_id=[0, 2], pad_token_id=3)

            def eval(self):
                return self

            def generate(self, **kwargs):
                tail = torch.full((1, kwargs["max_new_tokens"]), 1)
                return torch.cat([kwargs["input_ids"], tail], dim=1)

        runner = PolicyRunner(Model(), Tokenizer(), max_new_tokens=32,
                              generation_batch_size=1, device=torch.device("cpu"))
        trajectory = runner.rollout([(0, (1, 2, 3, 4))], repetitions=1,
                                    label="no-eos", do_sample=False)[0]
        self.assertEqual(trajectory.game.termination, "invalid:generation_limit")
        self.assertEqual(len(trajectory.decisions), 1)
        self.assertEqual(len(trajectory.decisions[0].completion_ids), 32)
        self.assertIsNone(trajectory.decisions[0].record()["termination_token_id"])

    def test_real_qwen_architecture_and_peft_update_on_cpu(self):
        """Small random architecture only: no pretrained weights or task experiment."""
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(17)
        tiny = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=128, attention_dropout=0.0,
            eos_token_id=2, pad_token_id=0,
        ))
        model = get_peft_model(tiny, LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
        model.enable_input_require_grads()
        runner = PolicyRunner(model, SimpleNamespace(eos_token_id=2), max_new_tokens=32,
                              generation_batch_size=2, device=torch.device("cpu"))
        transition = Game24([1, 2, 3, 4]).step("1 2 +")
        decisions = [
            Decision("a", "", [3, 4, 5], [6, 2], ["x", "EOS"], True, transition),
            Decision("b", "", [3, 5], [7, 8, 2], ["y", "z", "EOS"], True, transition),
        ]
        trajectories = [SimpleNamespace(trajectory_id="t", decisions=decisions)]
        _attach_old_and_reference_logprobs(runner, trajectories, score_batch_size=2)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        frozen_names = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
        config = ExperimentConfig.from_json(ROOT / "game24_experiment/experiment.json")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=config.learning_rate, weight_decay=config.weight_decay)
        advantages = SimpleNamespace(advantages={
            "a": SimpleNamespace(base=1.0, final=1.5),
            "b": SimpleNamespace(base=-0.5, final=-0.75),
        })
        with tempfile.TemporaryDirectory() as directory:
            metrics = _update(model, optimizer, trajectories, advantages, arm="two_step",
                              config=config, pad_id=runner.pad_id, device=runner.device,
                              microbatch_size=1, token_log_path=Path(directory) / "tokens.jsonl.gz")
        self.assertGreater(metrics["gradient_norm"], 0)
        self.assertEqual(metrics["generated_tokens"], 5)
        self.assertEqual(metrics["clipped_token_count"], 0)
        changed = {name for name, parameter in model.named_parameters()
                   if not torch.equal(parameter, before[name])}
        self.assertTrue(changed)
        self.assertFalse(changed & frozen_names)
        model.eval()
        with torch.no_grad(), model.disable_adapter():
            reference_after = score_logprobs(model, decisions, pad_id=0, device=runner.device)
        for decision, reference in zip(decisions, reference_after):
            self.assertTrue(torch.allclose(reference, torch.tensor(decision.reference_logp), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
