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
The RL-control benchmark is CPU-only and uses Gymnasium Acrobot-v1 and
LunarLander-v3 with a discretized Q-learning controller. It runs Random, TPE,
GPT-5.5 with context, and GPT-5.5 without context for 20 trials across seeds
`0..4`. The GPT-5.5 arms use high modeling effort and the last 5 trials of
history. The winning contextual arm disables explicit reasoning and qualitative
notes. It is strongest on both environment means in the committed run.
The credit-default benchmark is CPU-only and uses UCI dataset 350, Default of
Credit Card Clients (CC BY 4.0, DOI `10.24432/C55S3H`). The official archive
SHA-256, workbook schema, 60/20/20 stratified split, split seed, search space,
and 20-trial budget are pinned. All five optimizer seeds see the same train and
validation data; the held-out test split is evaluated only for each run's
validation-selected configuration.

Random, TPE, selected contextual GPT-5.5, and matched GPT-5.5/no-context
artifacts must all be present. Agent runs are fail-closed. The primary
trajectory metric is validation incumbent log loss; held-out test log loss is a
secondary generalization check. This is not a production credit-decision system
and must not be interpreted as one. The test split is reported after selection,
not used to select.

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
pip install -e ".[examples,ml,dev]"
pytest
python scripts/verify_classification_cumulative_error.py
python examples/hard_functions.py selfcheck
python examples/hard_functions.py plot
pip install -e ".[rl,examples]"
python examples/rl_control.py selfcheck
python examples/rl_control.py summary
python examples/rl_control.py plot
python examples/rl_control.py gif
python examples/credit_card.py selfcheck
python examples/credit_card.py summary
python examples/credit_card.py plot
python scripts/render_trajectory.py
```

Plotters must reject missing, stale, mixed-budget, or mixed-provenance data.
Keep exploratory outputs outside `docs/assets/`; only the complete canonical
run belongs in the documentation tree.

## Interpreting results

Report distributions or multi-seed means, not a best seed. State whether the
metric emphasizes early improvement or final quality. Agent latency and token
cost are separate from objective-evaluation cost and should be reported when
they materially affect the comparison.
