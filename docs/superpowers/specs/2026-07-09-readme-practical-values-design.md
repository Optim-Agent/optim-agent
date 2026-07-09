# README Practical Values Design

## Goal

Enrich `README.md` with a compact section that extracts the paper's core highlights and practical value for users evaluating `optim-agent`.

## Scope

Modify only `README.md`. Preserve the current README structure and benchmark sections. Do not rewrite the opening, change benchmark numbers, or add new experiments.

## Placement

Add a new section after `## Why optim-agent` and before the full documentation link.

## Section

Use the heading:

```markdown
## What you get in practice
```

The section should cover:

- small-budget leverage when objective evaluations are expensive and classical surrogates are data-starved;
- semantic tuning from parameter meanings plus trial history;
- auditable decisions through recorded context, proposals, outcomes, and optional rationales;
- bounded autonomy where the agent proposes, `optim-agent` validates, and the objective decides;
- drop-in deployment through the Optuna-style API, JSON/SQLite storage, local authenticated CLI backends, and zero runtime dependencies.

## Verification

Review the rendered Markdown shape by reading the edited section in context. No code tests are needed for a README-only copy change.
