# Local Reward Refinement for Long-Horizon Trajectories

[中文版](README.zh-CN.md) · [Original project ideas](idea.md)

This is a research project on **credit assignment in long-horizon language-agent trajectories**. When an agent receives a terminal outcome after many decisions, giving every decision the same trajectory-level signal can hide which local choices helped or hurt. We study how trajectories that revisit the same decision state can provide more local, action-level training signals. Code agents are a motivating application; the current implementations use controlled sequential tasks to examine the method first.

## Current method

The repository compares two methods under matched task protocols:

1. **Single-step action-mean GRPO baseline.** At a decision state, group observed trajectory visits by next action, average terminal rewards within each action, and normalize across the distinct action means with equal weight. The resulting state-action advantage is assigned to visits taking that action.
2. **Two-step gated refinement.** Start from visits whose first-action advantage has the smallest absolute magnitude. At their successor state, check whether the observed next actions have sufficient dispersion relative to the parent state and meet the method's action-coverage gate. For passing branches, add a first-step entry bonus to selected visits and strengthen the **signed** advantage of selected second actions. Other visits retain the single-step advantage; overlapping two-step chains follow the documented additive rule.

The [two-step specification](experiment/game24/two_step_gated_grpo_theory.md) and [baseline specification](experiment/game24/action_mean_grpo_baseline.md) define the selection quotas, gates, equations, overlap behavior, and policy objective precisely. Task-specific exceptions and settings are in the experiment protocols and configuration files. In the **current** implementation, an action or tool call is one decision step: its step advantage is shared by the model-generated tokens of that step, while policy ratios, clipping, and KL are computed per token. This is still **step-level credit assignment**, not a token-level reward-refinement algorithm.

## Experiments and current status

| Task | What is in the repository | First-run compute plan |
| --- | --- | --- |
| [Game of 24](experiment/game24/README.md) | Sequential arithmetic environment, official data with provenance, a locked plan, training code, tests, and a paired baseline/two-step protocol | One A100 80 GB GPU |
| [Six-block Blocksworld](experiment/BlocksWorld/README.md) | Sequential adaptation of PlanBench Blocksworld-4ops, symbolic environment, data-generation/audit instructions, training code, and paired protocol | One A100 80 GB GPU; Linux generator required |
| [Five-read HotpotQA](experiment/hotpotqa/README.md) | Selective-reading adaptation of HotpotQA distractor, eight-GPU training path, tests, and paired protocol | One Linux server with eight A100 GPUs |

These are first-run plans, not measured hardware guarantees. The Blocksworld and HotpotQA adaptations are separate interactive protocols, not official leaderboard scores. Local method and synthetic integration checks are described in the experiment guides. **No completed pretrained-model training comparison or empirical performance result is reported yet.** The [English method note](output/pdf/two_step_gate_grpo_en.pdf) and its [LaTeX source](paper/main.tex) describe the method, not experimental findings.

## Work in progress

- **Token-level variant:** We are considering a finer-grained version that would assign or refine credit within an action's generated tokens. Its objective, attribution rule, and evaluation protocol have not been fixed, and it is not part of the current experiments.
- **Compute and empirical evaluation:** We are currently looking for computing resources to run the planned pretrained-model comparisons and analyze gate behavior, performance, and actual compute cost.
- **Long-horizon agents:** After the controlled tasks, we hope to study whether the approach transfers to longer tool-use and code-agent trajectories. This remains a research direction.

## Discuss and collaborate

Feedback on the credit-assignment assumptions, experimental design, and failure cases is welcome. If you can share compute resources or are interested in running or extending an experiment, please [open a GitHub issue](https://github.com/ShuoZhang021022/local-reward-refinement-for-long-horizon-trajectories/issues) or email me at shuozhang2002@uchicago.edu to discuss the setup and required resources. The earlier bilingual project sketches are preserved as [idea.md](idea.md) and [idea.zh-CN.md](idea.zh-CN.md).
