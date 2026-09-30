# Game of 24 experiment

This directory contains the Game of 24 implementation of the two-step gated GRPO method and its single-step baseline. The current experimental choices are in [game24_experiment/experiment.json](game24_experiment/experiment.json), with the protocol and logging requirements in [game24_experiment_protocol_and_logging.md](game24_experiment_protocol_and_logging.md). The method documents are [two_step_gated_grpo_theory.md](two_step_gated_grpo_theory.md) and [action_mean_grpo_baseline.md](action_mean_grpo_baseline.md).

The unchanged source dataset and its provenance are in [data/](data/). See [game24_experiment/README.md](game24_experiment/README.md) for the implementation guide and all run details. Install a CUDA-enabled PyTorch suitable for the A100 and the dependencies in `game24_experiment/requirements.txt`, then run from this directory:

```bash
python -m unittest discover -s tests -v
python -m game24_experiment.train prepare --config game24_experiment/experiment.json --output runs/game24-plan-6xa100
python -m game24_experiment.launch_6gpu --plan runs/game24-plan-6xa100/plan.json --output-root runs/game24-6xa100
```

The launcher requires six distinct A100 40GB GPUs and starts one arm/seed run per GPU. The `locked_plan/` files are retained as a record of an earlier source revision. Their source hashes no longer match this code, so do not use that plan to run the current implementation; create a new plan with `prepare` as shown above. The `runs/` directory is ignored because it can contain large training artifacts. No pretrained-model training results are included in this upload.
