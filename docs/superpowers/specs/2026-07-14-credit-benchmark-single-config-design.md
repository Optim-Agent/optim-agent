# Credit Benchmark Single-Config Design

## Goal

Remove the obsolete credit-default effort-level comparison and publish only the
existing GPT-5.5 high-effort configuration, whose committed five-seed results
beat Random and TPE on validation and held-out test log loss.

## Changes

- Expose four methods: `Random`, `TPE`, `GPT-5.5`, and
  `GPT-5.5-no-context`.
- Rename the five `GPT-5.5-high` artifacts to `GPT-5.5` and delete the ten
  contextual `low` and `medium` artifacts.
- Rename the five no-context artifacts without changing their measured values.
- Configure both GPT-5.5 arms with `effort="high"`, `history=20`,
  `explicit_reasoning=True`, and `qualitative_notes=True`.
- Persist and validate those settings in benchmark artifacts and the manifest.
- Update the credit README table from the retained artifacts and regenerate
  `docs/assets/credit_card.png`.

## Constraints

- Keep the dataset, split, search space, seeds, trial count, and all measured
  objective values unchanged.
- Do not modify similarly named MNIST or CIFAR artifacts.
- Preserve unrelated worktree changes, including the current README edits.
- Use the existing artifact self-check and targeted test suite as verification.
