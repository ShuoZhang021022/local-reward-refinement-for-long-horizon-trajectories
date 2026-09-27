# Five-read HotpotQA comparison

This package implements the confirmed sequential-reading variant of the
[HotpotQA distractor data](https://hotpotqa.github.io/). It is not the official
leaderboard task: unread text is hidden until selected, and trajectories read
at most five of the ten paragraphs. See [PROTOCOL.md](PROTOCOL.md) for the
method and its limits.

Download the official labeled [train JSON](https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json)
and [distractor dev JSON](https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json).
Put them at the paths in `experiment.json` (or edit those paths before plan
creation). The official dev set is held out for final testing; 10% of the
official train set becomes validation. No answer or supporting fact is placed
in a model prompt. Use the pinned Python dependencies in
`../requirements.txt` plus a CUDA-enabled PyTorch build with
NCCL on a single Linux server with eight visible A100 GPUs. Preparation
rejects partial files: it expects
[90,447 train and 7,405 distractor dev questions](https://huggingface.co/datasets/hotpotqa/hotpot_qa).

From the repository root, prepare one immutable plan:

```powershell
python -m hotpotqa_experiment.train prepare --config hotpotqa_experiment/experiment.json --output runs/hotpotqa_plan
```

On that eight-A100 host, launch one process per GPU. Each command is one
joint eight-GPU training run, not eight independent runs. Both arms and all
three seeds use the **same** plan. For example:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 -m hotpotqa_experiment.train run --plan runs/hotpotqa_plan/plan.json --output runs/hotpotqa_baseline_20260926 --arm baseline --seed 20260926
torchrun --standalone --nnodes=1 --nproc-per-node=8 -m hotpotqa_experiment.train run --plan runs/hotpotqa_plan/plan.json --output runs/hotpotqa_two_step_20260926 --arm two_step --seed 20260926
```

The global budget remains 256 trajectories per question and eight questions
per optimizer update. Each GPU samples 32 trajectories per question with a
deterministic rank seed (`training_seed * 1009 + rank`), identically in both
arms. All 256 trajectories are grouped before anchor statistics and gates are
computed. Each GPU then computes its local trajectory-mean gradient; the
eight gradients are averaged before the same AdamW update. Validation and
final testing divide questions across GPUs and aggregate exact answer-F1
sums. The manifest records rank seeds, hardware, effective generation and
optimizer settings. Audit checkpoints include rank-0 trainable weights and
optimizer state plus RNG states from all eight ranks. The run refuses fewer
than eight local CUDA processes or non-A100 GPUs. Changes to source files
invalidate an existing locked plan; rerun `prepare` before formal training.

Repeat for seeds `20260927` and `20260928`. After all six runs complete,
compare them with `python -m hotpotqa_experiment.summarize`, passing each
directory as a separate `--run` and a JSON path as `--output`. The summary
checks the plan, complete run set, dev question coverage, and paired scores.

The plan and run manifests record effective settings, code hashes, data hashes,
split IDs, model/tokenizer revision, library versions, generation settings,
and trainable parameters. Rollout records keep each visible prompt, generated
selection, answer generation, token counts, and resulting reward. Training
records include anchor statistics, gate outcomes, update diagnostics, and
checkpoints. Every training decision is joined to its gate and advantage in
the trajectory log. `updates.jsonl` and the six-run summary report gate counts
and pass rates separately for selection steps 1–5, with explicit denominators
documented in `PROTOCOL.md`. Empty or truncated model outputs become zero-reward trajectories;
service or hardware failures stop the run. Answer tokens are logged but never
appear in the policy-loss decision list.

The confirmed budget is very large. With the official counts and 10% split,
there are 81,402 training questions and 9,045 validation questions. Across
both arms and three seeds, the one-epoch budget produces 125,033,472 training
trajectories. Each run has 10,176 updates and 1,019 full-validation snapshots
(including update zero and the final update), adding 55,301,130 validation
trajectories across six runs. Plan GPU time, disk, and checkpoint storage
accordingly. The repository excludes the HotpotQA data, and the eight-A100 training path
has not been run; there are no completed HotpotQA training results yet.
