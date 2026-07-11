# Benchmark contract

The committed benchmark artifacts support the claims in the README, docs, and
paper. They are evidence, not decorative assets. Tables and figures must be
generated from the JSON runs under `docs/assets/`.

## Suites

`manifest.json` records the stable suite identifiers, trial budgets, seeds,
result globs, and context policy. Individual result files remain authoritative
for backend, model, effort, search-space version, objective values, and sampled
parameters.

The hard-function comparison uses **no supplied task context**: generic
parameter names, bounds, and observed trial history only. A model may still
recognize a standard function from its bounds or values, so these runs measure
small-budget optimization rather than semantic-context benefit. Classification
runs provide the explicit with-context versus no-context comparison.

## Provenance requirements

Every agent result intended for publication must identify:

- suite and function or dataset;
- backend and exact model identifier;
- agent effort and context policy;
- trial budget, seed, and search-space version;
- ordered parameter proposals and objective values; and
- creation time or source commit when the runner provides it.

Baselines must use the same space, objective, budget, and seeds. A rotating
hosted model alias must be labeled as such; never silently present it as a
pinned model.

## Publication gate

Do not update prose or plots from a partial seed set. Before publishing:

```bash
pip install -e ".[examples,dev]"
pytest
python scripts/verify_classification_reward.py
python examples/hard_functions.py selfcheck
python examples/hard_functions.py plot
python scripts/render_trajectory.py
```

Plotters must reject missing, stale, mixed-budget, or mixed-provenance data.
Keep exploratory outputs outside `docs/assets/`; only the complete canonical
run belongs in the documentation tree.

## Interpreting results

Report distributions or multi-seed means, not a best seed. State whether the
metric rewards early improvement or final quality. Agent latency and token
cost are separate from objective-evaluation cost and should be reported when
they materially affect the comparison.
