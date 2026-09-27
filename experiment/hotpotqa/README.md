# Five-read HotpotQA experiment

This folder contains a self-contained paired comparison of the action-mean
GRPO baseline and the two-step gated method on a selective-reading variant of
HotpotQA distractor. The model initially sees ten paragraph titles, reads at
most five paragraphs one at a time, and may submit an answer early. This is an
interactive adaptation, not an official HotpotQA leaderboard submission.

Start with the [experiment guide](hotpotqa_experiment/README.md) and
[locked-method protocol](hotpotqa_experiment/PROTOCOL.md). The executable
first-run settings are in
[hotpotqa_experiment/experiment.json](hotpotqa_experiment/experiment.json).
The shared algorithms are specified in the [two-step method](two_step_gated_grpo_theory.md)
and [single-step baseline](action_mean_grpo_baseline.md). The
game24_experiment/ Python package bundles the shared runtime imported by
this experiment; the Game24 task itself is not part of the HotpotQA evaluation.

Run commands from **this directory** on one Linux server with eight A100 GPUs.
Install a CUDA-enabled PyTorch build with NCCL and the packages in
[requirements.txt](requirements.txt).
Place the official labeled train and distractor-dev JSON files under data/
as described in [data/README.md](data/README.md). Then run:

    python -m unittest discover -s tests -v
    python -m hotpotqa_experiment.train prepare --config hotpotqa_experiment/experiment.json --output runs/hotpotqa_plan

Use the same prepared plan for both arms and all three seeds. Each run jointly
uses all eight GPUs, with 32 of the 256 trajectories per question on each GPU;
anchor statistics and gates use the full 256 before synchronized updates.
The experiment guide gives the `torchrun` and summary commands. Recreate any
older plan because the source hashes have changed. Dataset files and run artifacts are
excluded from this upload. No pretrained-model training run, final comparison,
or official HotpotQA joint score is included. The local tests use synthetic
questions and do not establish an empirical result.
