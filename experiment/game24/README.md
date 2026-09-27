# Game of 24 experiment

This directory contains the Game of 24 implementation of the two-step gated GRPO method and its single-step baseline. The experimental choices are in [game24_experiment/experiment.json](game24_experiment/experiment.json), with the protocol and logging requirements in [game24_experiment_protocol_and_logging.md](game24_experiment_protocol_and_logging.md). The method documents are [two_step_gated_grpo_theory.md](two_step_gated_grpo_theory.md) and [action_mean_grpo_baseline.md](action_mean_grpo_baseline.md).

The unchanged source dataset and its provenance are in [data/](data/). The implementation guide is [game24_experiment/README.md](game24_experiment/README.md). A previously prepared plan and split are in [locked_plan/](locked_plan/); its model revision and all 16 source hashes were checked against this upload. The plan's recorded Git status describes the original preparation workspace.

From this directory, install a CUDA-enabled PyTorch suitable for your GPU and the dependencies in game24_experiment/requirements.txt, then run:

```bash
python -m unittest discover -s tests -v
python -m game24_experiment.train run --plan locked_plan/plan.json --output runs/baseline-20260926 --arm baseline --seed 20260926
python -m game24_experiment.train run --plan locked_plan/plan.json --output runs/two-step-20260926 --arm two_step --seed 20260926
```

Follow the implementation guide for the other two seeds and summarization. To create a fresh plan instead, run `python -m game24_experiment.train prepare --config game24_experiment/experiment.json --output runs/game24-plan`; this resolves and pins the model revision again. The runs/ directory is ignored because it can contain large training artifacts. No pretrained-model training results are included in this upload.