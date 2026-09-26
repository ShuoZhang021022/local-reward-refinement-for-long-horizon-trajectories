# Local Reward Refinement for Long-Horizon Trajectories

[中文版](README.zh-CN.md)

## Mathematical method PDF

- [English method note](output/pdf/two_step_gate_grpo_en.pdf)

The PDF describes the current two-step gated GRPO method through its update rule. It contains sections 1–6 and no experimental results.

This project explores an extension of GRPO for long-horizon trajectories, especially code agents. The goal is to redistribute reward signals across steps so that gradient updates better reflect the contribution of each step.

1. **State-based reward refinement.** For a fixed state, collect all historical trajectories that pass through it. Examine the token entropy of the next-step choices already made at that state, then use the graph structure and relative relationships among trajectories to refine the local reward signal.

2. **Advantage-guided exploration.** Apply a similar idea to the advantage of each next-step action. A large absolute advantage suggests a clearer effect, but may also mean that the action has already been explored well. An advantage close to zero gives a weaker signal, yet may leave room for further discovery. Start new trajectories from such actions and explore more deeply.
