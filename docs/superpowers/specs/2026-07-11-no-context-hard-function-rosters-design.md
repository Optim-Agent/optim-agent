# No-Context Hard-Function Agent Rosters Design

## Goal

Refresh the Branin and Ackley-5D benchmark as a no-context comparison of
top-tier coding agents, rotating free OpenCode agents, Optuna TPE, and uniform
random search. Every agent run uses medium reasoning effort, five seeds, and a
ten-trial budget.

## Benchmark Contract

The hard-function benchmark deliberately withholds task semantics. Agent
samplers receive only generic parameter names (`x1`, `x2`, ...), numeric bounds,
and observed trial history. They must not receive the function name, optimum,
functional form, or a study-wide context string.

All agent samplers use:

- `context=None`
- `effort="medium"`
- `n_init=3`
- ten trials per function
- seeds `0, 1, 2, 3, 4`
- the current `AgentSampler` prompt construction and reply-validation path

Random and TPE remain non-agent baselines and therefore have no model or
reasoning-effort setting.

## Pinned Rosters

The top-tier roster is:

| Published label | Backend | Model argument |
|---|---|---|
| GPT-5.5 | `codex` | `gpt-5.5` |
| Opus-4.8 | `claude` | `claude-opus-4-8` |
| Sonnet-5 | `claude` | `claude-sonnet-5` |
| GLM-5.2 | `opencode` | `opencode-go/glm-5.2` |

The free OpenCode roster available on 2026-07-11 is:

| Published label | Backend | Model argument |
|---|---|---|
| Big-pickle | `opencode` | `opencode/big-pickle` |
| DeepSeek-V4-Flash | `opencode` | `opencode/deepseek-v4-flash-free` |
| Nemotron-3-Ultra | `opencode` | `opencode/nemotron-3-ultra-free` |
| MiMo-v2.5 | `opencode` | `opencode/mimo-v2.5-free` |

The free roster is intentionally explicit rather than discovered dynamically.
OpenCode's free pool rotates, so a future refresh may replace unavailable
entries, but a single benchmark run must never substitute a model silently.

Random and TPE belong to both plotting groups. The four top-tier agents belong
only to the tier plot, and the four free agents belong only to the free plot.

## Preflight And Failure Behavior

Before deleting the published curves or starting the full matrix, execute one
small structured-output request against every configured agent model with
medium effort. Each smoke test must prove that the CLI accepts the exact backend,
model, and effort combination and returns parseable JSON.

If any smoke test fails, stop before the full run and report the unavailable
model. Do not replace it, change its model ID, or fall back to a backend default.
The benchmark sampler retains its existing per-trial random fallback for
transient failures, but full-run logs must be monitored and any fallback warning
must make the run invalid for publication.

## Implementation Shape

`examples/hard_functions.py` remains the single benchmark entry point. Its pool
will contain ten methods: the eight agent models plus Random and TPE. The
distributed command continues to run five seeds concurrently within one method
and methods sequentially, limiting simultaneous nested CLI calls to five.

The agent curve path passes the pool's backend and explicit model to
`AgentSampler` with medium effort and `context=None`. No alternate
`no_context` branch or contextual hard-function candidate remains.

Each JSON artifact records:

- published label
- backend
- exact model argument
- `effort="medium"` for agents and `null` for baselines
- `no_context=true` for agents
- seed and ten-trial budget
- ten values and ten parameter dictionaries for both Branin and Ackley-5D

The expected publication output is 50 JSON files: five seeds for each of ten
methods. Plot loading rejects missing, extra, stale, or provenance-incompatible
hard-function files.

## Plots And Documentation

Generate two plots:

- `docs/assets/hard_benchmarks_tier.png`: four top-tier agents, Random, and TPE
- `docs/assets/hard_benchmarks_free.png`: four free agents, Random, and TPE

Both titles state that the agents use medium effort with no task context and
that curves average five seeds. Stable dimensions, readable legends, and the
same Branin/Ackley axes are retained for comparison.

`README.md` and `docs/index.html` will:

- state that all hard-function agent runs are no-context
- list the exact top-tier and free rosters
- identify GPT-5.5 as an explicitly pinned Codex model
- explain that the free OpenCode pool rotates and may require a future refresh
- emphasize that the free comparison requires no paid model API
- report metrics derived from the newly validated JSON files
- show both regenerated plots

The paper remains unchanged because it documents a separate legacy experiment.

## Testing And Verification

Tests are written before implementation and cover:

- exact labels, backends, model IDs, and plot groups
- explicit `gpt-5.5` pinning
- medium effort and `context=None` for every agent sampler
- provenance fields in written JSON
- rejection of wrong model, effort, context mode, trial count, seed set, function
  set, or unexpected label
- tier/free plot membership with Random and TPE in both

After the run, verification requires:

1. 50 compatible JSON files and no other `hard_curves_*` files.
2. Five seeds per method and ten values/parameter dictionaries per function.
3. Exact model and backend provenance for all eight agent methods.
4. Medium effort and `no_context=true` for all agent artifacts.
5. Successful numerical self-check and relevant automated tests.
6. Visual inspection of both plots for clipping, overlap, missing series, and
   incorrect legends.
7. README and HTML metrics reproduced directly from the validated JSON.

## Resource Expectation

With three random warmup trials and seven agent-guided trials per function, the
eight agent models require approximately 560 nested agent calls:

`8 models x 5 seeds x 2 functions x 7 calls = 560 calls`.

The run is therefore intentionally preceded by strict smoke tests and executed
in visible foreground batches so model errors, timeouts, or random fallbacks are
detected promptly.
