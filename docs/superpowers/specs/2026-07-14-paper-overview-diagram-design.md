# Paper Overview Diagram Design

## Goal

Redesign Figure 1 so a reader without HPO or LLM-agent background can understand
the contribution before reading the Method section. The figure should show a
clear before/after transition: optim-agent automates the repetitive proposal
work previously done by a human while preserving human control over the task and
mechanical control over valid settings.

## Composition

Keep the existing 750-by-224-point, full-width aspect ratio so the figure does
not consume additional page space. Use two main panels joined by a transformation
arrow.

### Before: Manual Tuning

- A human-brain icon represents the expert.
- The loop reads `Choose Settings` -> `Run Experiment` -> `Read Score` -> back
  to the expert.
- A short panel message states that the human repeats every proposal step.

### Transformation

- A central arrow connects the panels.
- Its label is `Automate Repeated Proposals`.
- The arrow communicates transfer of repetitive work, not removal of the human.

### With optim-agent: Guarded Automation

- The human supplies `Goal + Parameter Meanings + Limits` once.
- An electric-brain icon labeled `AI Agent` proposes the next settings.
- A green `Typed Safety Check` gates every proposal before execution.
- The user's experiment returns a score, which is stored in history and guides
  the next AI proposal.
- A compact failure branch states `Invalid: Retry Once, Then Valid Random`.
- The human input does not enter the feedback loop again; the objective and
  declared limits remain authoritative.

## Visual Language

- Use editable TikZ vector artwork; do not add a generated bitmap dependency.
- Use plain-language labels in the diagram and reserve mathematical notation for
  the caption and body text.
- Depict the human and AI with distinct circular icons: warm orange for human
  judgment and blue with circuit details for the AI agent.
- Use charcoal for experiment execution, green for validation and results, and
  red only for the invalid-proposal branch.
- Keep a white or near-white background, strong arrow contrast, and labels large
  enough to survive full-width AAAI scaling.
- Avoid decorative art, nested panels, and labels that explain implementation
  details irrelevant to a first-time reader.

## Backup and Integration

Before editing, preserve the current assets as:

- `paper/src/figures/optim_agent_overview_original.tex`
- `paper/src/figures/optim_agent_overview_original.pdf`

Then update the existing editable source and vector output in place:

- `paper/src/figures/optim_agent_overview.tex`
- `paper/src/figures/optim_agent_overview.pdf`

Update the Figure 1 caption in `paper/src/main.tex` to explain the before/after
boundary without repeating every on-figure label. Do not alter unrelated paper
content unless pagination requires removing prose that the new figure directly
duplicates.

## Caption

Use this meaning, adjusting wording only for clarity and line wrapping:

> From manual tuning to guarded agent tuning. A human normally repeats the
> propose--evaluate--interpret loop. With optim-agent, the human declares the
> objective, parameter meanings, and legal search space once; a CLI agent then
> proposes settings from context and trial history. Typed validation gates every
> proposal before the user-owned experiment runs, and observed scores guide the
> next trial. Invalid proposals retry once and otherwise fall back to valid
> random sampling.

## Verification

1. Confirm both original backup files exist and are nonempty before replacing
   the active asset.
2. Compile the standalone TikZ source with PDFTeX and inspect the rendered
   figure for label fit, arrow direction, and visual hierarchy.
3. Compile the complete manuscript with PDFTeX after the caption update.
4. Require exactly seven main-content pages with `References` beginning page 8.
5. Require no overfull boxes, undefined references, or missing assets.
6. Render and inspect the full paper and a high-resolution Figure 1 crop.
7. Save the verified manuscript as `paper/main.pdf`.
