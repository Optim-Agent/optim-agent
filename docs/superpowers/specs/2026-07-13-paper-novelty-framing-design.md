# Paper Novelty Framing Design

## Goal

Make the paper's novelty immediately legible as a semantics-aware HPO systems
contribution without claiming that LLM-based HPO, contextual candidate
generation, faithful explanation, robustness, or generalization is already
established as new or superior.

## Claim Hierarchy

The paper will lead with the following claim:

> optim-agent makes parameter semantics a first-class input to a conventional
> HPO study while keeping the agent behind typed validation, bounded fallback,
> ask--tell and define-by-run APIs, interchangeable authenticated CLI backends,
> and persistent study records.

The novelty lies in this combination and software boundary. It does not lie in
using an LLM for optimization by itself, training a new optimizer, inventing a
new surrogate, or obtaining faithful explanations from model rationales.

## Formal Distinction

Let `C` contain study-level and per-parameter semantic descriptions. A
context-blind optimizer is invariant to those descriptions:

```text
p(x_{t+1} | X, D_t, C) = p(x_{t+1} | X, D_t, C')
```

for any descriptions `C` and `C'` when the numerical space and history are
fixed. optim-agent intentionally removes this invariance by conditioning its
proposal policy on `C`, while the deterministic validator still maps every
proposal into the declared numerical or categorical space.

This distinction will appear as one unnumbered display because it is the
conceptual center of the contribution, not an auxiliary metric. The paper will
describe semantic advantage as a testable hypothesis, not as an established
causal result, because the current vision comparison also changes proposal
availability and the effort-matched credit comparison is non-significant.

## Manuscript Changes

### Title and Abstract

- Change the title to `Optim-Agent: Validated Semantics-Aware Hyperparameter
  Optimization with CLI Agents`.
- Open the abstract with the semantic blindness of conventional black-box HPO.
- State the four-part contribution compactly: semantic schemas, standard study
  API, validated agent boundary, and auditable storage.
- Retain the repository URL and current empirical caveats.

### Introduction and Contributions

- Name the missing abstraction: ordinary samplers consume coordinates and
  outcomes but not what a coordinate means.
- Explain that optim-agent preserves the objective as authoritative and changes
  only the proposal policy.
- Replace the contribution bullets with:
  1. a semantics-aware sampler contract for mixed HPO spaces;
  2. a reusable, backend-agnostic and validated CLI-agent implementation;
  3. an auditable evaluation that exposes both gains and confounds.
- Use a qualified combination claim rather than an absolute first/only claim.

### Related Work

- Keep the acknowledgement that LLM-based HPO is not itself new.
- Distinguish optim-agent from trained meta-optimizers, LLM components embedded
  in Bayesian optimization, full-pipeline AutoML agents, and prompt-specific
  optimizers.
- State that the contribution is the reusable boundary between these agents and
  an ordinary HPO study, not prompt engineering.

### Method

- Introduce the context-blind invariance definition before the agent policy.
- Define the semantic schema as part of the study state and proposal contract.
- Connect typed validation directly to the novelty: semantic reasoning may
  influence which point is proposed but cannot redefine the legal search space.

### Discussion and Conclusion

- Explain why semantic conditioning is useful in small-budget, interpretable
  spaces and potentially harmful when semantic priors are wrong.
- Describe records as auditable semantic traces, not faithful explanations.
- Conclude with the new systems boundary and retain the non-superiority caveat.

## Claims to Avoid

- `optim-agent is the first or only contextual HPO method`.
- `LLM proposals are explainable` or model rationales are faithful.
- `semantic proposals are more robust or general` without cross-task evidence.
- `context caused the vision improvements` under the current startup confound.
- `optim-agent is generally superior to TPE` given the credit results.

## Page and Verification Constraints

- Replace existing prose instead of adding a novelty section or comparison
  table.
- Keep exactly seven pages of main content, with References beginning on page 8.
- Do not change margins, fonts, spacing, or figure scales to recover space.
- Run pdfTeX after every manuscript iteration.
- Require no overfull boxes, unresolved citations, or undefined references.
- Render and inspect all pages, with focused inspection of the title/abstract,
  related-work transition, formal definition, and page 7 boundary.
- Copy the verified PDF to `paper/main.pdf`.

## Acceptance Criteria

1. A reviewer can identify the novelty from the title, abstract, first two
   introduction pages, and contribution bullets without inferring it.
2. The context-blind versus semantics-aware distinction is formally stated.
3. Prior LLM-HPO work is acknowledged and differentiated on concrete system
   properties.
4. The paper uses `auditable` rather than unsupported `explainable` claims.
5. Existing empirical caveats and statistical conclusions remain unchanged.
6. The final rendered PDF satisfies all page and layout constraints.
