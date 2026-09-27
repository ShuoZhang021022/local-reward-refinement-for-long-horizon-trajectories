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
`../requirements.txt` plus a CUDA-enabled PyTorch build on the
training host. Preparation rejects partial files: it expects
[90,447 train and 7,405 distractor dev questions](https://huggingface.co/datasets/hotpotqa/hotpot_qa).

From the repository root, prepare one immutable plan:

```powershell
python -m hotpotqa_experiment.train prepare --config hotpotqa_experiment/experiment.json --output runs/hotpotqa_plan
```

On a CUDA training host, run both arms under each confirmed seed with the
**same** plan. For example:

```powershell
python -m hotpotqa_experiment.train run --plan runs/hotpotqa_plan/plan.json --output runs/hotpotqa_baseline_20260926 --arm baseline --seed 20260926
python -m hotpotqa_experiment.train run --plan runs/hotpotqa_plan/plan.json --output runs/hotpotqa_two_step_20260926 --arm two_step --seed 20260926
```

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
accordingly. This repository does not contain the HotpotQA data or a
completed model run.
