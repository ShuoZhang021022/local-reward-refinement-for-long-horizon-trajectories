"""CPU checks for the eight-rank aggregation used by HotpotQA training."""

from __future__ import annotations

import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from hotpotqa_experiment.distributed import (
    SynchronizedOptimizer, _join_gzip_shards, _merge_update_metrics, rank_seed,
)


class _FakeDist:
    class ReduceOp:
        SUM = "sum"

    def __init__(self, summed_gradient: list[float]):
        self.summed_gradient = summed_gradient

    def all_reduce(self, tensor: torch.Tensor, *, op: str) -> None:
        assert op == self.ReduceOp.SUM
        tensor.copy_(torch.tensor(self.summed_gradient, dtype=tensor.dtype))

    def get_world_size(self) -> int:
        return 8


class HotpotDistributedTests(unittest.TestCase):
    def test_global_gradient_mean_applied_before_optimizer_step(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        synchronized = SynchronizedOptimizer(
            optimizer, [parameter], _FakeDist([16.0, 24.0]))
        parameter.grad = torch.tensor([4.0, 9.0])
        synchronized.step()
        self.assertTrue(torch.allclose(parameter.detach(),
                                       torch.tensor([0.8, -1.3])))
        self.assertAlmostEqual(synchronized.gradient_norm, 13 ** 0.5)

    def test_global_update_metrics_weight_token_counts(self) -> None:
        parts = [{"objective": 1.0, "loss": -1.0, "generated_tokens": 2,
                  "clipped_token_count": 1, "mean_token_kl": 0.5,
                  "trajectory_count": 4, "decision_count": 6,
                  "learning_rate": 1e-5}]
        parts += [{**parts[0], "objective": 0.0, "loss": 0.0,
                   "generated_tokens": 4, "clipped_token_count": 0,
                   "mean_token_kl": 0.25} for _ in range(7)]
        merged = _merge_update_metrics(parts, gradient_norm=3.0)
        self.assertEqual(merged["trajectory_count"], 32)
        self.assertEqual(merged["decision_count"], 48)
        self.assertEqual(merged["objective"], 0.125)
        self.assertEqual(merged["clipped_token_fraction"], 1 / 30)
        self.assertAlmostEqual(merged["mean_token_kl"], 8 / 30)
        self.assertEqual(merged["gradient_norm"], 3.0)

    def test_rank_seeds_do_not_overlap_between_confirmed_run_seeds(self) -> None:
        derived = [rank_seed(seed, rank)
                   for seed in (20260926, 20260927, 20260928)
                   for rank in range(8)]
        self.assertEqual(len(derived), len(set(derived)))

    def test_gzip_shards_merge_in_rank_order(self) -> None:
        with TemporaryDirectory() as root:
            directory = Path(root)
            shards = []
            for rank in range(8):
                shard = directory / f"rank{rank}.gz"
                with gzip.open(shard, "wt", encoding="utf-8") as stream:
                    stream.write(f"{rank}\n")
                shards.append(shard)
            destination = directory / "merged.gz"
            _join_gzip_shards(destination, shards)
            with gzip.open(destination, "rt", encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "".join(f"{rank}\n" for rank in range(8)))
            self.assertTrue(all(not path.exists() for path in shards))


if __name__ == "__main__":
    unittest.main()
