# Paper Overview Diagram Design

## Goal

Create an original three-panel overview figure for the optim-agent AAAI paper,
inspired by the comparative role of AgentHPO Figure 1 but not by its artwork.
The figure must make optim-agent's semantics-aware, validated systems boundary
understandable before the reader reaches the Method section.

## Composition

The figure is a horizontal, full-width comparison with three equal panels and
subtle separators. Each panel shows the same proposal--evaluation loop at a
different system boundary.

### Panel A: Human Tuning

- Inputs: parameter meanings, task knowledge, and observed results.
- Actor: human expert.
- Action: manual configuration proposal.
- Feedback: experiment result returns to the expert.
- Message: semantic reasoning is available, but the loop is manual.

### Panel B: Context-Blind HPO

- Inputs: typed numerical space `X` and trial history `D_t`.
- Actor: Random/TPE numerical sampler.
- Action: candidate `x_{t+1}`.
- Feedback: objective value returns to numerical history.
- Semantic schema `C` is shown outside the sampler with a clear `not consumed`
  marker.
- Message: numerical HPO is intentionally invariant to parameter descriptions.

### Panel C: optim-agent

- Inputs: semantic schema `C`, typed space `X`, and history `D_t`.
- Proposal stage: replaceable CLI agent, labeled `Codex / Claude / OpenCode`.
- Boundary stage: JSON proposal `x_tilde` passes through typed validator `V`.
- Execution stage: the user-owned objective `f(x)` evaluates the legal point.
- Storage stage: outcome is recorded in JSON/SQLite and fed into the next trial.
- Failure branch: invalid responses retry once and then use valid random
  fallback.
- Message: semantics may influence the proposal, but not the legal domain or
  objective.

## Visual Language

- White background suitable for AAAI print and grayscale reproduction.
- Flat vector style with square or lightly rounded boxes; no copied robot or
  human artwork from AgentHPO.
- Charcoal text and borders.
- Orange identifies semantic inputs and human knowledge.
- Blue identifies proposal and validation stages.
- Green identifies objective outcomes and persisted history.
- Dashed gray treatment identifies unavailable or rejected paths.
- Arrows are thick enough to remain legible at two-column paper width.
- Typography is sans serif within the asset, with no text smaller than the
  effective AAAI caption size after scaling.
- Mathematical labels use ASCII-compatible approximations where necessary in
  SVG and are explained in the caption.

## Asset and Paper Integration

- Create editable source: `paper/src/figures/optim_agent_overview.svg`.
- Export vector PDF: `paper/src/figures/optim_agent_overview.pdf`.
- Add one `figure*` after the Introduction contribution bullets.
- Caption the distinction without claiming that model rationales are faithful
  explanations.
- Remove the existing `hard_benchmarks_tier.png` / `hard_benchmarks_free.png`
  figure block from the manuscript; retain the analytic results table and all
  analytic discussion.
- The new overview becomes Figure 1, while classification and credit figures
  renumber automatically.

## Caption

Use this content, adjusting only for line wrapping:

> Three HPO boundaries. Human tuning uses semantics but requires manual
> proposals. Random and TPE consume the typed space and numerical history but
> not parameter descriptions. optim-agent makes descriptions first-class inputs
> to a replaceable CLI-agent proposal policy, then mechanically validates the
> JSON candidate before the user objective executes it; outcomes persist to the
> next trial, and invalid replies use bounded retry and valid random fallback.

## Page-Budget Strategy

The overview replaces the existing full-width analytic-curves figure rather
than adding another large float. If pagination still changes, recover space only
by deleting repeated prose that the overview now communicates. Do not alter
margins, fonts, spacing, table sizes, or the remaining figure scales.

## Verification

1. Render the standalone SVG and PDF and inspect labels, arrow direction,
   contrast, and grayscale legibility.
2. Compile with pdfTeX after the figure is inserted.
3. Require exactly seven main pages and References as the first line on page 8.
4. Require no overfull boxes, undefined references, or missing assets.
5. Render all paper pages and inspect the new Figure 1 at full-page and cropped
   resolution.
6. Confirm that the overview is visually original and does not reuse AgentHPO
   icons, layout details, or textual labels beyond the general three-way
   comparison concept.
7. Copy the verified output to `paper/main.pdf`.
