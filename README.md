# Local Reward Refinement for Long-Horizon Trajectories

[简体中文](README.zh-CN.md)

## Project blueprint

This project explores an extension of Group Relative Policy Optimization (GRPO) for long-horizon trajectories, especially those produced by code agents. The central question is whether trajectory-level feedback can be assigned more meaningfully to individual steps, so that policy updates receive a more informative learning signal.

This repository currently contains a research blueprint, not an implemented algorithm or an experimental result.

## Motivation

In a long code-agent run, many decisions precede the final outcome. A trajectory-level reward alone may give little guidance about which decisions helped, hurt, or deserve further exploration. The proposed direction is to use the history of trajectories passing through a state, together with relationships among their continuations, to refine local reward or advantage signals.

## Idea 1: History-aware local reward refinement

For a given state, collect historical trajectories that pass through it and examine their next-step choices. Study the token entropy of those choices as a possible measure of local uncertainty or diversity. Represent shared states, next-step choices, and subsequent outcomes with a graph; use its structure and relative comparisons among continuations to investigate how trajectory feedback might be reassigned across steps.

The intended result is a candidate local learning signal for GRPO, with its exact definition and policy-update rule still to be specified. In particular, the entropy source, state-matching criterion, graph construction, and reward redistribution rule are open research decisions.

## Idea 2: Advantage-guided exploration

The same historical and relational view could be applied to advantages associated with next-step actions. A large absolute advantage may identify an action whose observed effect is comparatively clear, while also suggesting that this branch has already been explored substantially. An advantage near zero gives a weak directional signal; it may be a useful candidate for deeper branching and new trajectories if its uncertainty or coverage indicates that more exploration could help.

These interpretations are hypotheses to test. A near-zero advantage could also reflect genuinely similar outcomes or noisy estimates, so its magnitude alone should not be treated as proof of unexplored potential. The project will investigate whether combining advantage magnitude with coverage and uncertainty can guide where to allocate additional exploration.

## Research questions to resolve before implementation

1. What constitutes a state and a step in a code-agent trajectory, and when should two states be considered the same?
2. Is next-step token entropy measured from the policy distribution, empirical choices in stored trajectories, or another distribution?
3. Which nodes, edges, and relative outcome comparisons define the graph?
4. How are local rewards or advantages computed from the graph, and how do they enter the GRPO objective? Are trajectory-level totals preserved?
5. Which parameters receive gradients, and how are historical trajectories sampled or refreshed?
6. What criteria trigger deeper exploration, and what budget, depth, and failure-handling rules apply?
7. Which tasks, data splits, baselines, and evaluation measures will test credit assignment and long-horizon code-agent performance?

## Proposed next milestones

- Specify the state, action, reward, and advantage definitions, then write the objective and update procedure.
- Define an exploration policy and comparison protocol without changing the research question through hidden implementation defaults.
- Build a small diagnostic study to inspect entropy, graph relationships, and advantage estimates before full training.
- Compare against a clearly specified GRPO baseline on long-horizon code-agent tasks.

**Status:** Concept stage. No training procedure, hyperparameters, or empirical claims are fixed yet.
