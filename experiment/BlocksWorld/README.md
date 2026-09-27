# Six-block PlanBench Blocksworld experiment

This folder contains a self-contained implementation of a paired comparison
between the action-mean GRPO baseline and the two-step gated method. It adapts
PlanBench's Blocksworld-4ops problems to **one model action per interaction**.
The result is a new sequential protocol, not an official PlanBench score.

Start with the [English protocol and run guide](blocksworld_experiment/README.md).
The [method specification](blocksworld_experiment/PROTOCOL.md) explains the
advantage equations, selected first-run settings, evaluation, and logging.
The executable settings are in
[`blocksworld_experiment/experiment.json`](blocksworld_experiment/experiment.json).

The `blocksworld_experiment/` package implements data verification, the
symbolic environment, training, and paired summarization. Its shared GRPO
method, token objective, model-scoring, and checkpoint functions are bundled
as the unchanged `game24_experiment/` runtime copied from this repository's
Game24 experiment. The two Blocksworld test modules are in `tests/`.

Run commands from **this directory**. Install a CUDA-enabled PyTorch build
appropriate for the GPU and the packages in [`requirements.txt`](requirements.txt).
The original PlanBench generator must also be available on Linux at the
pinned revision shown in the run guide. Before using the GPU, generate the
500 six-block PDDL problems, run the shortest-plan audit, and prepare the
locked experiment plan. Then run baseline and two-step for each of the three
seeds and produce the paired summary. Generated data and run artifacts belong
under the ignored `data/` and `runs/` directories; preserve them separately
because the upstream generator has no seed argument.

No official six-block dataset, pretrained-model training run, or success-rate
comparison is included in this upload. Local CPU tests use synthetic
instances; they do not establish a result for the 500-problem experiment.
