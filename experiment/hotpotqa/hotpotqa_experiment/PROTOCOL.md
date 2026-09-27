# HotpotQA sequential-reading comparison: working protocol

**Status:** The research-critical settings below reflect the user's confirmed
choices. This is a new interactive variant of HotpotQA distractor, not an
official HotpotQA leaderboard protocol. Formal training requires the official
labeled files and a CUDA training host; they are not present locally.

## Authority and task boundary

- **User-confirmed interaction:** Each question has ten candidate paragraphs.
  One model decision reads one previously unread paragraph or submits early.
  A trajectory may read at most five; immediately after the fifth read, the
  model is asked for a final answer with no additional selection decision.
  Before that limit, submitting is an alternative to reading another
  paragraph. The baseline and two-step arm share the same action and read
  budgets.
- **User-confirmed visibility:** Unread paragraphs display only their stable
  one-based ID and original title. Their sentence text is hidden until read.
  The selection model receives a textual rendering of the entire structured
  action/observation history plus the current state. It returns a structured
  next-step choice. The answer prompt also uses the trajectory's visible
  history, including an explicit final `submit` action when the model submits
  early; the fifth read requests an answer without a separate submit action.
  The prompt never includes gold answer or gold supporting facts.
- **User-confirmed grouping:** The anchor ignores order and earlier prompts.
  Within one question it contains the set of read paragraph IDs and the IDs,
  titles, and unread status of the remaining paragraphs. The question ID is
  included to prevent cross-question grouping. Full prompt text and anchor
  are logged separately. Two trajectories that read the same IDs in a
  different order share an anchor but need not have identical prompts or
  conditional action probabilities. This is an anchor-state abstraction,
  following the grouping *idea* of GiGPO, not an assertion that all prompts
  under an anchor are identical. See https://arxiv.org/pdf/2505.10978 .
- **User-confirmed answer model:** Each arm uses its own current model to
  generate the final answer, both after early submit and after the fifth
  read. Answer-generation tokens are excluded from the policy objective.
  Reading-policy updates still change shared model weights, so answer
  generation can change indirectly. The complete answer prompt and generated
  answer are logged for that reason.
- **User-confirmed reward:** Terminal reward is the official HotpotQA
  *answer-only F1* rule. Report answer EM and answer F1 separately. Gold
  supporting-fact coverage by the read paragraphs is diagnostic only; it
  is not the official supporting-fact or joint score. Official scorer:
  https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py .
- **User-confirmed failure rule:** A malformed selection, duplicate read,
  out-of-range paragraph ID, or invalid final answer ends that trajectory
  immediately with answer F1=0; no model-output retry or sample skip occurs.
  Service and hardware failures abort the run separately. An empty or
  whitespace-only answer and either generation that reaches 32 tokens without
  EOS are invalid. A nonempty answer with explanatory text is scored directly
  by official answer F1, rather than stripped or rewritten.
- **User-confirmed data roles:** Split the labeled official train file into
  training and validation; use the entire labeled official dev file only for
  the final test. The official blind test file has no reference answers and is
  outside this experiment. Reserve 10% of official train with seed 20260926;
  the implementation records all question IDs and uses `round` for the one
  fractional-record boundary. File loading checks the full expected train and
  distractor-dev counts (90,447 and 7,405) before preparing or running.
- **User-confirmed first-run transfer:** Both arms use the Qwen3-4B checkpoint
  and the previously confirmed Game24 algorithm coefficients, LoRA scope,
  optimizer, learning rate, clip and KL settings, and initial frozen reference
  policy. The user further confirmed one epoch, 256 trajectories per question,
  eight questions per update, validation every ten updates, three seeds
  (20260926/27/28), and a 32-token cap separately for choices and answers.
  Both generations sample at temperature=1, top_p=1, top_k=0 during training.
  Evaluation is one greedy trajectory per question, and the highest validation
  answer F1 chooses the checkpoint; ties choose the earliest update.
- **Previously confirmed methods:** Both arms group visits by the same anchor,
  compute action mean terminal reward and equally weighted across-action
  standard deviation, and use the same generated-token objective. The
  baseline uses only state action-mean advantage. The two-step arm also uses
  the existing first-stage 20% selection, at least five *observed distinct
  legal* successor actions, child dispersion and delta gate, lambda and beta
  corrections, and overlap rule. See the shared
  [two-step method](../two_step_gated_grpo_theory.md) and
  [single-step baseline](../action_mean_grpo_baseline.md). Answer tokens
  are outside the visits passed to these calculations.

## Decision states and transitions

For `k` previously read paragraphs with `0 <= k < 5`, the selectable keys are
`read:ID` for every unread ID plus `submit`. There are `11-k` selectable keys
before observing policy rollouts. In particular, after four reads there are
six remaining reads plus submit, or seven keys. `submit` requests an answer
and has no successor *selection* state. Reading the fifth paragraph requests
an answer without a new selection action, so its child state is also absent
for the two-step gate. Neither a legal menu size nor the existence of a
successor alone satisfies the gate: it requires at least five distinct legal
actions actually observed at the same child anchor in the old rollout batch.

Zero-read `submit` is allowed. The code uses one JSON object for each selection, either
`{"action":"read","paragraph_id":N}` or `{"action":"submit"}`. This
structure and the shortest-answer-only prompt are ordinary interface choices,
not claims about the official HotpotQA format. The fifth read and early
submit both call the same arm's current model through the same answer
generation pathway.

## Evaluation and interpretation

The official distractor task gives all ten paragraphs at once and reports
answer and supporting-fact metrics; the proposed selective reveal changes
the information available to the agent. See https://hotpotqa.github.io/ .
The final report must identify this as a five-read interactive variant.

For both arms, log per question and trajectory: read IDs and order, full
selection and answer prompts, generated selections and answer, answer EM/F1,
gold-support document coverage, early versus forced submission, read count,
generation calls/tokens, anchor key at every choice, counts of observed legal
actions, parent and child dispersions, gate outcomes and advantages. Compare
arms on identical question splits and starting weights. Distinguish the
number of *available* choices from the number of *observed* distinct actions
and from the number of gate passes. Because the answer model is the arm's
updated model, a change in answer score alone cannot be attributed solely to
changed reading order.

Each training `rollouts_*.jsonl.gz` line is one complete trajectory. Its
decisions contain the selection step, full visible prompt, generated tokens,
action and child anchor, advantage, and gate result where that action was a
first-stage candidate. `stats_*.json` preserves all anchor and candidate-gate
records. Each `updates.jsonl` row includes `gate_by_step` for selection steps
1–5, where step 1 means no text has been read yet. The primary
`candidate_gate_pass_rate` is passing candidate `(anchor, action)` pairs divided
by all first-stage candidate pairs at that step, including terminal actions.
`nonterminal_candidate_gate_pass_rate` instead divides by candidate pairs
with a successor selection anchor. `selected_first_visit_gate_pass_rate`
divides selected trajectory visits whose parent gate passed by all first-stage
selected visits at that step. A missing denominator produces `null`, not zero.
The baseline computes these gate rates only as counterfactual diagnostics;
its `lambda_applied_visits` and `beta_applied_visits` remain zero. Reading the
fifth text and `submit` have no successor selection anchor, so they cannot
pass the two-step gate. The final six-run summary recomputes per-step rates
from summed counts rather than averaging per-update percentages.

## Execution prerequisites

Place the official labeled train JSON and distractor dev JSON at the paths in
`experiment.json`, or edit those paths before `prepare`. `prepare` locks file
hashes, question IDs, model and tokenizer revision, effective config, and
source hashes. Run the three seeds for both arms against the same plan on a
CUDA host. Model and service failures abort the run with failure recorded.
No full local run or research conclusion follows from unit tests alone.
