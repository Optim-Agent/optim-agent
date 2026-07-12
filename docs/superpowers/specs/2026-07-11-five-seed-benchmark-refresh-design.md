# Five-Seed Benchmark Refresh Design

## Goal

Refresh the published MNIST, CIFAR-10, and hard-functions benchmarks with five
seeds and a consistent four-candidate comparison:

- Random
- TPE
- GPT-5.5 medium
- GPT-5.5 medium without context

Every Codex-backed run must explicitly select `gpt-5.5` and medium reasoning.
Classification uses 10 trials per seed and reports the early-optimization
cumulative best-so-far error `sum(best_test_error_so_far[i] for i in 1..N)`,
where lower is better.

## Execution

Reuse the five-seed scheduling and GPU-rotation pattern already implemented by
`scripts/verify_classification_cumulative_error.py`. Existing fresh Random, TPE, and
context-enabled GPT classification curves may be reused. Rerun the missing
no-context GPT classification curves for seeds 0 through 4.

Add the smallest practical distributed interface to
`examples/hard_functions.py`: it runs seeds 0 through 4 concurrently for the
same four candidates. Each GPT candidate uses the Codex backend with model
`gpt-5.5`, medium effort, and differs only in whether study and parameter
context is supplied. Hard-functions results continue to use the existing 10
trial budget and objective definitions.

A failed seed must make the orchestration command fail. Plot generation starts
only after all expected JSON curve files exist and pass their existing shape
and metadata checks.

## Artifacts and Plots

Store reproducible per-seed JSON curves and regenerate:

- `docs/assets/mnist_benchmarks.png`
- a CIFAR-10 benchmark plot in `docs/assets`
- the hard-functions benchmark plot in `docs/assets`

Each refreshed plot contains only the four approved candidates and labels the
five-seed aggregation. Metrics in prose and tables are calculated from these
JSON artifacts rather than transcribed from console output.

## Documentation

Replace stale benchmark material in `README.md` and `docs/index.html` with the
new trial counts, five-seed protocol, 16-dimensional classification spaces,
explicit GPT-5.5 medium selection, distributed reproduction commands,
cumulative-error definition, and generated results. Preserve unrelated installation, API, and
usage documentation. Rewrite the whole README only if the resulting changes
exceed 30 percent of the file.

## Verification

Run the repository tests and each example self-check. Then verify that every
dataset and method has exactly five curves, all classification curves contain
10 trials, plot legends contain only the four approved candidates and show five
seeds, and GPT metadata records `gpt-5.5` with medium effort. Recompute the
published aggregates directly from the generated JSON and inspect the plots for
legibility.
