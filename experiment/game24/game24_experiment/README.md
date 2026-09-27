# Game of 24 training implementation

This is the first runnable implementation of the local [two-step method](../two_step_gated_grpo_theory.md) and [single-step baseline](../action_mean_grpo_baseline.md) for Game of 24. The user authorized choosing the remaining parameters. The choices are recorded in [`experiment.json`](./experiment.json); they are **experimental choices**, not settings supplied by ToT or proven optimal values. The external A100 has not yet been connected, so no Qwen training result exists.

## Protocol

The model sees a constant system instruction and only the **current numbered multiset** in each user prompt. It emits one response in `left_slot right_slot operator` format, such as `2 1 -`; this is one model generation and one environment operation. The next prompt is rebuilt from the new numbers, with no previous calls or original puzzle. The tokenizer may split the response into any number of tokens; every generated token, including an emitted EOS, receives the same step advantage. Prompt tokens and environment observations receive no training advantage.

All arithmetic uses exact rational numbers. Three legal operations must leave 24; a model-generated invalid action or a response without EOS by the 32-token limit immediately ends the trajectory with reward 0. The entire EOS list and padding ID come from the locked checkpoint's generation configuration. The current official checkpoint lists EOS IDs `151645` and `151643`; either ends the response, and that emitted token remains a training target. Padding after EOS is excluded. `prepare` locks these IDs, and `run` checks them again. Infrastructure exceptions abort the run and are not turned into model failures. `INVALID` is a single action group within a state for action-mean statistics, but is excluded from the five-legal-action gate and second-action selection. Different numbered slots remain different actions even when they hold equal values; `+` and `*` commute for the **same two slots**, while `-` and `/` retain order.

## Fixed first-run settings

| Choice | Value | Reason for first run |
| --- | --- | --- |
| Model | `Qwen/Qwen3-4B-Instruct-2507`, exact revision locked at `prepare` | Non-thinking 4B model with trainable weights; fits the action-only protocol. [Model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| Trainable scope | LoRA on `q/k/v/o/gate/up/down_proj`, rank 16, alpha 32, dropout 0 | Makes one A100 80GB plausible while keeping both arms' update scope identical. No extra dropout regularizer. |
| Reference | Frozen initial base model with LoRA adapter disabled | Same reference for both arms; no second 4B model copy in GPU memory. [PEFT API](https://huggingface.co/docs/peft/package_reference/peft_model) |
| Gate/advantage | `ω=1`, `λ=0.5`, `β=1.5` | Requires child action-mean dispersion at least as large as parent; modest positive entry bonus and second-action scaling. These are fixed choices, not tuned claims. |
| Objective | Per-token `ε=0.2`, `κ=0.01`; exact step/token/trajectory reduction in the method documents | KL is enabled in both arms, with the same reference and coefficient. |
| Optimizer | AdamW, learning rate `1e-5`, weight decay 0, no gradient clipping, one update per on-policy batch | Avoids extra regularization or multiple passes over stale rollouts. |
| Training budget | One pass through 1136 training puzzles, 8 puzzles/update, 256 trajectories/puzzle, 3 paired seeds | Repeated rollouts per puzzle give the five-observed-action gate a chance to fire; 142 updates per seed and arm. |
| Generation | Sample from raw policy: temperature 1, top-p 1, top-k 0; max 32 new tokens | Old-policy token logprobs match the behavior distribution without sampling truncation. |
| Evaluation | Greedy, one complete attempt per puzzle; validate initially, every 10 updates, and at end; choose highest validation accuracy, earliest update on ties | Stable checkpoint selection without looking at test. Test the selected checkpoint once at the end. |
| Split | Official zero-based indices `[900,1000)` held for test; each other original-rank block of 100 contributes 10 validation puzzles, final block of 62 contributes 6 | Fixed 1136/126/100 train/validation/test split. SHA256 ordering and seed `20260926` determine IDs; the plan saves the full lists. |

The three training seeds are `20260926`, `20260927`, and `20260928`. The two arms use the same plan, data, initial revision, seeds, optimizer, prompting, attempted trajectories per puzzle, and evaluation protocol. Each arm collects rollouts from **its own current old policy**. Early invalid termination can make the actual model-call and generated-token counts differ between arms; both are logged. The sampled actions are not constrained or repaired. This is a deliberately simple first run; a zero-success batch can yield zero policy advantage. Training logs expose that limitation rather than silently changing reward or sampling.

## External GPU commands

The repository includes the unchanged official CSV at [`../data/24.csv`](../data/24.csv). Its trusted upstream commit and SHA256 are recorded in [data/README.md](../data/README.md) and checked against fixed constants before parsing or splitting. A reordered or modified CSV is rejected, even if it still has 1362 unique puzzles and the expected puzzle at index 900. Git newline conversion is disabled for this file. To restore the exact source file on another host:

```bash
mkdir -p data
curl -L https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/733b009f627f8e5c81c3e5461391d3aa3e0dd18f/src/tot/data/24/24.csv -o data/24.csv
```

Install a CUDA-enabled PyTorch suitable for the GPU, then install `game24_experiment/requirements.txt`. Run the unit tests before the experiment:

```bash
python -m unittest discover -s tests -v
python -m game24_experiment.train prepare --config game24_experiment/experiment.json --output runs/game24-plan
python -m game24_experiment.train run --plan runs/game24-plan/plan.json --output runs/baseline-20260926 --arm baseline --seed 20260926
python -m game24_experiment.train run --plan runs/game24-plan/plan.json --output runs/two-step-20260926 --arm two_step --seed 20260926
```

Repeat the two `run` commands for seeds `20260927` and `20260928`, using separate output directories. `prepare` requires network access to resolve the model's immutable revision and read its generation configuration. It does not load the 4B weights or sample trajectories. Model and tokenizer use the same locked repository revision. The same source CSV and root working directory must be used for every run. Plans use schema version 2; older plans must be recreated. A run refuses to overwrite a nonempty output directory and preserves its existing manifest. If an infrastructure failure interrupts a run, it is incomplete and must be rerun under a new directory; there is no automatic retry or resume.

After all six runs finish, summarize the paired test outcomes:

```bash
python -m game24_experiment.summarize --output runs/game24-summary.json \
  runs/baseline-20260926 runs/two-step-20260926 \
  runs/baseline-20260927 runs/two-step-20260927 \
  runs/baseline-20260928 runs/two-step-20260928
```

The summary requires all arm/seed combinations in the locked plan (six runs for this configuration), matching library versions, and an unaltered copy of the same plan. It checks the training order, every update record, the SHA256 and byte size of each rollout/token/state-statistics artifact and audit checkpoint, the scheduled validation traces and metrics, the earliest-best selection, saved adapters, and the selected test trace and metrics. Missing even a complete seed pair is rejected. This verifies recorded artifacts and their internal consistency; it does not independently prove that the model produced their contents. It reports per-seed paired differences and a puzzle-bootstrap interval. That interval conditions on the three recorded training seeds; it does not claim to quantify uncertainty over all possible training seeds.

The report also includes per-run and per-arm actual generation-decision calls, generated tokens, invalid rates, candidate/passed gate actions, and eligible/applied λ and β visits. A generation-decision call is one model response for one Game24 state, including responses batched together; reference/current scoring forwards are not included. Gate fractions use candidate actions as the denominator. Applied λ/β fractions use all training decisions as the denominator. Baseline gate counts are diagnostics on its own trajectories; its applied λ/β counts remain zero. Summarization reads and hashes every audit checkpoint, so allow time for the full six-run audit.

Each run writes a manifest with config, trusted data provenance, source hashes, package/runtime versions, GPU details, trainable parameter names, resolved LoRA settings, and every effective optimizer parameter-group setting (including AdamW defaults such as betas and epsilon). It writes full rollout and evaluation traces, action/state statistics, every candidate parent/child gate, per-visit advantages, and per-generated-token old/current/reference logprobs, ratios, clipping flags, and KL values. The test set is evaluated only after validation has selected a checkpoint.

## Per-state counts and training diagnostics

For each update, `stats_XXXX.json` contains `states[].gate_summary`:

| Field | Counting unit / interpretation |
| --- | --- |
| `visit_count`, `first_selected_visit_count` | All visits to this state, and visits selected by the first 20% rule |
| `candidate_action_count` | Distinct parent actions selected for gate checks |
| `passed_action_count` | Candidate parent actions passing all gates |
| `passed_candidate_action_fraction` | Passed actions / candidate actions; `null` if no candidate exists |
| `passes_five_action_count`, `passes_delta_action_count` | Each individual gate among the same candidate actions |
| `gate_reason_counts`, `parent_zero_variance` | Failed-candidate reasons, and the zero-parent-variance case that creates no candidates |
| `lambda_applied_visit_count` | Visits at this state actually receiving the first-position bonus |
| `beta_applied_visit_count` | Visits at this state actually selected for second-action modification, based on their own preceding gate |
| `overlap_applied_visit_count` | Visits in both roles |
| `nonzero_applied_advantage_visit_count` | Visits whose actual policy advantage is nonzero |

The `*_eligible_visit_count` fields describe the two-step calculation on the batch. Baseline logs explicitly label it `counterfactual_diagnostic_only`; its applied modification counts are zero and `advantages[].applied_advantage` is always the baseline advantage. `advantages[].final` retains the computed two-step value for comparison and is not the baseline's training coefficient. State logs also list same-successor action groups with sample counts and the number of statistical actions observed only once.

`updates.jsonl` and console output flag zero-success batches, zero policy advantages, and (for the two-step arm) no passing gate. These are diagnostics on the existing training batch. They do not add sampling, retries, dense reward, a new gate, or early stopping. Gate passage alone is not evidence that a low-sample action mean is reliable. With one optimizer step per rollout batch, ratios are normally near one during gradient evaluation, so clipping is expected to be mostly inactive.

## Audit checkpoints

`checkpoints/state_0000.pt` and one file after every update save all trainable parameter tensors, optimizer state, Python/Torch/CUDA random states, and update/plan identifiers. They are taken after any scheduled validation and identify the policy before the next batch. The fixed base weights are recovered from the pinned model revision. `checkpoints.jsonl` records each file's SHA256 and byte count; update logs link the previous and new checkpoint. Best and final adapters remain available separately.

This retains 143 audit snapshots per run for the current configuration. Full optimizer states increase disk use substantially; `manifest.json.estimated_audit_checkpoint_bytes` estimates the run's snapshot storage after the first update, excluding rollouts and token logs. Preserve sufficient disk space on the external host. These audit snapshots do not introduce automatic continuation of failed experiments.

The current Windows workspace has CPU PyTorch. Tests cover both checkpoint EOS IDs, strict source checks, complete-run aggregation and tamper rejection, per-state counts, checkpoint restoration, the installed Transformers chat-template return type, and a small randomly initialized Qwen3 architecture with real PEFT generation and gradient updates. These tests download no model weights and make no Game24 performance claim. Actual pretrained 4B generation, A100 memory/speed, and learning remain unverified until the external host is available. No experiment result is claimed by this repository yet.
