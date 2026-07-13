# RL Control Benchmark Design

## Goal

Add a CPU-only reinforcement-learning hyperparameter benchmark that demonstrates
optim-agent on Acrobot-v1 and LunarLander-v3, with committed result artifacts,
a README figure/table, and an optional LunarLander policy GIF if local rendering
is reliable.

## Scope

The benchmark will live in one script, `examples/rl_control.py`, and cover:

- `Acrobot-v1`
- `LunarLander-v3`
- `Random`
- `TPE`
- `GPT-5.5`
- `GPT-5.5 w/o context`

The benchmark will commit JSON result artifacts under `docs/assets/`, render a
single figure `docs/assets/rl_control.png`, add a compact README section, and
record the suite in `benchmarks/manifest.json`.

## Approach

Use Gymnasium environments and a small in-script discretized Q-learning
evaluator. This avoids adding Stable-Baselines3 or PyTorch as benchmark
requirements while still tuning real RL behavior. The tuned parameters are:

- learning rate
- discount factor
- epsilon decay
- minimum epsilon
- discretization bins
- training episodes

Each trial trains with a fixed CPU budget and evaluates the learned policy over
fixed evaluation episodes. The scalar objective is mean evaluation return; higher
is better. For plotting, each seed contributes an incumbent best-so-far curve.

## Context Policy

The contextual GPT-5.5 condition receives task and parameter descriptions,
including environment semantics and the exploration/exploitation tradeoff. The
no-context condition receives only parameter names, bounds, and trial history.
Random and TPE remain unchanged baselines.

## Artifacts

Expected artifact shape:

- `docs/assets/rl_control_Random_s0.json`
- `docs/assets/rl_control_TPE_s0.json`
- `docs/assets/rl_control_GPT-5.5_s0.json`
- `docs/assets/rl_control_GPT-5.5-no-context_s0.json`
- repeated for the committed seed set
- `docs/assets/rl_control.png`
- optional `docs/assets/lunarlander_policy.gif`

The JSON files will include schema version, method, backend, model, context
policy, seed, trials, environment names, objective name, values, params, and
best values.

## README

Add a section after the image-classification benchmark:

- state the benchmark is CPU-only and uses Gymnasium control tasks
- show `rl_control.png`
- include one table with Acrobot and LunarLander mean final returns
- describe the improvement conservatively, based on committed artifacts only
- include reproduction commands

No broad claim about RL performance or policy quality will be made; this is an
HPO/sample-efficiency demonstration.

## Optional GIF

Generate a LunarLander GIF only if local Gymnasium Box2D rendering works without
fragile system dependencies. If rendering fails, skip the GIF and keep the figure
and table as the committed visual evidence.

## Testing

Tests will cover:

- benchmark constants and method contract
- parameter validation and objective direction
- artifact validation rejects malformed JSON
- summary/table values are computed from committed artifacts
- renderer emits `rl_control.png`
- README and manifest reference the RL suite

The runnable selfcheck will avoid real agent calls and avoid long environment
training.
