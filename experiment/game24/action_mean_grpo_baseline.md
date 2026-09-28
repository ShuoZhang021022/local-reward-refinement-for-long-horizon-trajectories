# Single-Step Action-Mean GRPO Baseline

Status: confirmed comparison method. The Game24 implementation and CPU checks are complete; training the actual 4B model on an A100 has not yet occurred. This method's training advantage uses only action-group statistics at the current step, without a gate correction based on the next action $a'$. Policy updates follow the confirmed per-token procedure in the [two-step method](./two_step_gated_grpo_theory.md), including the same per-token KL term. See the [Game24 protocol](./game24_experiment_protocol_and_logging.md) and [configuration](./game24_experiment/experiment.json) for its complete settings and INVALID special case.

## 1. Relationship to the two-step method

Confirmed shared rules: One edit or tool call is one decision step. The old policy samples complete trajectories and terminal rewards. At the same complete decision state, terminal rewards are first averaged separately for each distinct next action, and those action means receive equal weight in normalization. All model-generated tokens of a step share its advantage; each token has its own policy ratio and clipping. The objective is averaged over tokens within each step, then over steps within each trajectory, and then over trajectories. Tool results, environment observations, and prompts are context only, not prediction targets carrying the step advantage.

Confirmed baseline difference: Each step trains directly with its action-mean-normalized advantage. The first-stage selection of the 20% closest-to-zero advantages belongs only to the two-step method; this baseline neither performs that selection nor adds $\lambda$. It therefore does not use next-action $a'$ statistics at child state $(s,a)$, the five-distinct-action requirement, the $\Delta$ gate, $A_2$, $\beta$, second-stage action selection, or the overlap formula for adjacent two-step chains. Its step advantage is always $A^{\mathrm{state}}$.

## 2. Trajectories, states, and action grouping

Under the same task and initial conditions, collect $N$ complete trajectories $\tau_i$ with the old policy $\pi_{\mathrm{old}}$. Each trajectory has $T_i$ decision steps and terminal reward $R_i$. State $s_{i,j}$ is the complete decision state at step $j$ of trajectory $i$, and $a_{i,j}$ is that step's edit or tool-call action. State equivalence must follow the two-step method; it cannot be loosened or tightened only for the baseline.

For each decision position, define the observed visits and distinct actions at state $s$ as

$$
\mathcal I(s)=\{(i,j):s_{i,j}=s\},\qquad
\mathcal U(s)=\{a_{i,j}:(i,j)\in\mathcal I(s)\},\qquad
d_s=|\mathcal U(s)|.
$$

For each $u\in\mathcal U(s)$, group the visits that choose $u$ and calculate their mean terminal reward:

$$
\mathcal I(s,u)=\{(i,j)\in\mathcal I(s):a_{i,j}=u\},\qquad
n_s(u)=|\mathcal I(s,u)|,\qquad
q_s(u)=\frac{1}{n_s(u)}\sum_{(i,j)\in\mathcal I(s,u)}R_i.
$$

This averages over observed visits within an action, rather than placing individual trajectory rewards directly into the cross-action standard deviation. Counting per visit is confirmed: If a trajectory revisits exactly the same state, each visit is a separate decision record. If those visits choose the same action, that trajectory's terminal reward $R_i$ appears repeatedly in the sum for $q_s(u)$. This weights by visit count; it does not make the visits statistically independent.

## 3. Equal-weight normalization of action means and step advantages

Give each of the $d_s$ distinct actions at state $s$ equal weight when computing the mean and standard deviation of their action means:

$$
\bar q_s=\frac{1}{d_s}\sum_{u\in\mathcal U(s)}q_s(u),\qquad
\sigma_{q,s}=\sqrt{\frac{1}{d_s}\sum_{u\in\mathcal U(s)}\bigl(q_s(u)-\bar q_s\bigr)^2}.
$$

Assign the resulting normalized action advantage to every observed visit that chose that action:

$$
\boxed{
A^{\mathrm{base}}_{i,j}=A^{\mathrm{state}}_{i,j}=
\begin{cases}
\dfrac{q_{s_{i,j}}(a_{i,j})-\bar q_{s_{i,j}}}{\sigma_{q,s_{i,j}}},
&\sigma_{q,s_{i,j}}>0,\\[6pt]
0,&\sigma_{q,s_{i,j}}=0.
\end{cases}}
$$

Visits at the same state choosing the same action share an advantage even when their trajectories have different terminal rewards. If only one distinct action was observed, or all action means are equal, then $\sigma_{q,s}=0$ and the visit's advantage is zero; the entire trajectory is not discarded. Distinct actions have equal weight in normalization, but every sampled visit remains in the policy objective. No additional $1/n_s(u)$ factor or other correction for repeated action counts is applied. These rules match $A^{\mathrm{state}}$ in the two-step method.

## 4. Broadcasting to tokens and updating the policy

Suppose the model generates $L_{i,j}$ tokens $y_{i,j,1:L_{i,j}}$ at step $(i,j)$, including any reasoning text and edit or tool-call content generated at that step. All of those tokens use the same step advantage:

$$
A^{\mathrm{base}}_{i,j,k}=A^{\mathrm{base}}_{i,j},\qquad k=1,\ldots,L_{i,j}.
$$

Let $h_{i,j,k}$ be the complete context before predicting a token. The per-token policy ratio and clipped term are

$$
r_{i,j,k}(\theta)=
\frac{\pi_\theta(y_{i,j,k}\mid h_{i,j,k})}
{\pi_{\mathrm{old}}(y_{i,j,k}\mid h_{i,j,k})},\qquad
\ell_{\mathrm{clip}}(r,A)=
\min\!\left\{rA,\operatorname{clip}(r,1-\epsilon,1+\epsilon)A\right\}.
$$

Only model-generated tokens contribute to $L_{i,j}$ and its sum. As in the two-step method, per-token KL is enabled, with the same reference policy $\pi_{\mathrm{ref}}$ and coefficient $\kappa>0$. Use the per-token KL estimator from [original GRPO](https://arxiv.org/html/2402.03300):

$$
D^{\mathrm{KL}}_{i,j,k}(\theta)=
\frac{\pi_{\mathrm{ref}}(y_{i,j,k}\mid h_{i,j,k})}
{\pi_\theta(y_{i,j,k}\mid h_{i,j,k})}
-\log\frac{\pi_{\mathrm{ref}}(y_{i,j,k}\mid h_{i,j,k})}
{\pi_\theta(y_{i,j,k}\mid h_{i,j,k})}-1.
$$

Within each step, average the policy clipped term and KL term over that step's tokens; then average over steps and trajectories. The baseline objective is

$$
\boxed{
J^{\mathrm{base}}(\theta)=
\mathbb E_{\{\tau_i\}\sim\pi_{\mathrm{old}}}
\left[
\frac{1}{N}\sum_{i=1}^{N}\frac{1}{T_i}\sum_{j=1}^{T_i}
\frac{1}{L_{i,j}}\sum_{k=1}^{L_{i,j}}
\left(
\ell_{\mathrm{clip}}\!\left(r_{i,j,k}(\theta),A^{\mathrm{base}}_{i,j}\right)
-\kappa D^{\mathrm{KL}}_{i,j,k}(\theta)
\right)
\right].}
$$

Maximize $J^{\mathrm{base}}$, or minimize its negative, to update the parameters through the current policy's token ratios and per-token KL term. Advantages and grouping statistics are computed from old trajectories and held fixed during the update. Per-token KL is confirmed. The first Game24 run uses $\kappa=0.01$, $\epsilon=0.2$, and the frozen initial model as reference policy. Both arms share the same reference policy, KL definition, coefficients, and reduction scheme.

## 5. Procedure and comparison boundary

1. Collect trajectories with the same old policy, task, sampling budget, and terminal-reward definition as the two-step method.
2. At each identical complete state $s$, group visits by current action $u$ and compute each mean terminal reward $q_s(u)$.
3. Give distinct action means equal weight to compute $\bar q_s$ and $\sigma_{q,s}$, then assign $A^{\mathrm{base}}_{i,j}$ to every visit.
4. Broadcast each step advantage to all tokens generated by the model at that step; compute a separate policy ratio and clipping term for each token.
5. For each generated token, compute both the clipped term and per-token KL term. Average over generated tokens within a step, then over steps and trajectories, and update the policy.

Here, "ignoring $a'$" means not explicitly grouping, selecting, gating, or correcting advantages by the next action or child state. Because $q_s(u)$ still comes from complete-trajectory terminal reward $R_i$, later behavior inevitably affects it indirectly. A signal wholly independent of later outcomes would require a different reward definition and would be a different baseline.

For a controlled comparison, both arms should share terminal reward, state matching, trajectory sampling, token scope, clipping, KL, loss reduction, and evaluation. The baseline uses only each step's $A^{\mathrm{state}}$; the two-step method additionally applies near-zero selection, next-action gating, and advantage corrections.

## 6. Task-specific parameters and implementation status

- Game24's reference policy, coefficients, data split, sampling budget, failure handling, stopping condition, and evaluation metrics are fixed in its protocol and configuration. The baseline may store counterfactual diagnostics for the two-step rule on its trajectories, but updates use only $A^{\mathrm{state}}$; logs distinguish calculated quantities from those actually applied. Applying this method to another task requires confirming that task's settings separately.
