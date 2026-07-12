<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/optim-agent-logo-dark.svg">
    <img alt="optim-agent" src="docs/assets/optim-agent-logo-light.svg" width="500">
  </picture>
</p>

<h1 align="center">optim-agent</h1>

<p align="center">
  <strong>Agentic system optimization with coding agents.</strong><br>
  Automate the iterative parameter-tuning work of an algorithm engineer.
</p>

<p align="center">
  <a href="https://pypi.org/project/optim-agent/"><img alt="PyPI" src="https://img.shields.io/pypi/v/optim-agent"></a>
  <a href="https://pypi.org/project/optim-agent/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/optim-agent"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/pypi/l/optim-agent"></a>
  <img alt="Zero runtime deps" src="https://img.shields.io/badge/runtime%20deps-0-brightgreen">
</p>

<p align="center">
  <strong>English</strong> |
  <a href="docs/i18n/README_ZH.md">简体中文</a> |
  <a href="docs/i18n/README_JA.md">日本語</a> |
  <a href="docs/i18n/README_KO.md">한국어</a> |
  <a href="docs/i18n/README_FR.md">Français</a> |
  <a href="docs/i18n/README_DE.md">Deutsch</a> |
  <a href="docs/i18n/README_ES.md">Español</a> |
  <a href="docs/i18n/README_PT.md">Português</a> |
  <a href="docs/i18n/README_RU.md">Русский</a>
</p>

optim-agent uses Claude Code, Codex, or OpenCode to optimize any system that
exposes **configurable parameters** and a **measurable objective**. It combines
what each parameter *means* with what the trial history *shows*, then proposes
the next configuration to evaluate. Your objective remains the authority:
optim-agent proposes, validates, records, and safely falls back when an agent
reply is invalid.

| Models | Systems | Research |
|---|---|---|
| Training, architecture, and RL experiments | Inference, latency, cost, control, and decision rules | Quant signals, simulations, and scientific workflows |

<p align="center"><a href="#install"><code>pip install optim-agent</code></a></p>

## Why optim-agent

- **Semantic proposals** - coding agents reason over parameter meanings, study
  context, and observed outcomes instead of treating every dimension as an
  anonymous coordinate.
- **Small-budget leverage** - useful when evaluations are expensive and classical
  surrogates are still data-starved.
- **Auditable decisions** - JSON/SQLite studies retain configurations,
  outcomes, states, context, and optional agent rationale.
- **Bounded execution** - the agent only proposes values; optim-agent validates
  them against the declared space, and invalid output falls back to safe
  sampling.

## Install

```bash
pip install optim-agent
```

Plus at least one agent CLI on your PATH, already authenticated:
[claude](https://docs.anthropic.com/en/docs/claude-code),
[codex](https://github.com/openai/codex), or
[opencode](https://github.com/sst/opencode).

## Quickstart

```python
import optim_agent as oa

def objective(trial):
    threshold = trial.suggest_float(
        "threshold", 0.05, 0.95,
        context="decision threshold; higher values trade recall for precision",
    )
    budget = trial.suggest_int(
        "budget", 10, 200, log=True,
        context="compute or operating budget; larger values may improve quality",
    )
    return evaluate_system(threshold=threshold, budget=budget)  # your code

study = oa.create_study(
    direction="maximize",
    sampler=oa.AgentSampler(
        backend="claude",  # or "codex" / "opencode"
        effort="high",
        context="maximize system quality under a strict operating-cost budget",
    ),
    storage="study.json",  # optional: persist and resume
)
study.optimize(objective, n_trials=20)
print(study.best_value, study.best_params)
```

`context` is optional but powerful: it tells the agent what the system and its
parameters represent, so it can reason like an algorithm engineer rather than
a blind point-picker. Set it study-wide on `AgentSampler(context=...)`,
per-parameter on each `suggest_*(..., context=...)`, or both.

## Where it applies

| Area | Parameters optim-agent can tune | Example objective |
|---|---|---|
| **Model training** | learning rates, architectures, augmentation, regularization | validation quality, compute, robustness |
| **Inference and serving** | quantization, batching, decoding, caching, routing | quality, latency, throughput, cost |
| **Quantitative research** | signal windows, thresholds, rebalance rules, risk controls | walk-forward return, drawdown, turnover |
| **Reinforcement learning and decisions** | objective weights, exploration schedules, environment settings, policy thresholds | return, safety, sample efficiency |
| **Scientific workflows** | simulation inputs, solver settings, experimental controls | fit, error, runtime, resource use |
| **Black-box systems** | any bounded categorical, integer, or continuous configuration | any scalar score you can measure |

For reinforcement learning, optim-agent tunes the system around the learning
loop; it does not replace the policy-learning algorithm.

## Watch the search unfold

![Agent optimization trajectory compared with TPE](docs/assets/optimization_trajectory.gif)

This seed-0 Branin trace shows where TPE and GPT-5.5 sample under the same
10-trial budget, alongside the incumbent objective after each trial. It is a
trajectory illustration, not the aggregate benchmark; the multi-seed results
and reproduction commands are below. Regenerate it from committed JSON with
`python scripts/render_trajectory.py`.

**Full documentation:** [docs/index.html](docs/index.html) - served as a
website via GitHub Pages (Settings > Pages > deploy from branch, `main` /`docs`).

## Benchmarks: agents vs. TPE and random search

The classification comparisons use the same four candidates: **Random**,
Optuna **TPE**, **GPT-5.5 w/ context**, and **GPT-5.5 w/o context**. Each curve
is the mean of **5 fresh seeds** (`0..4`) at **10 trials**. Both GPT-5.5
conditions explicitly pin `gpt-5.5` with medium reasoning effort
(`model_reasoning_effort=medium`); no CLI-default model is used. **GPT-5.5 w/
context** additionally receives natural-language text descriptions of the
study objective and all 16 parameters, while **GPT-5.5 w/o context** receives
none of those descriptions.

For classification, the primary metric emphasizes fast improvement:

```text
cumulative_best_so_far_error = sum(best_test_error_so_far_at_i for i in 1..10)
```

Lower is better. Both runs receive the declared parameter bounds, but only the
context-enabled run receives the study and parameter text descriptions. Neither
run uses hand-picked anchors.

This is an end-to-end context comparison, not a wording-only ablation. Without
an initial schema, `n_init=4` and four concurrent workers make trials 0–6 use
the same parameter proposals as Random; GPT-5.5 controls only trials 7–9. The
context-enabled cumulative-error path receives the schema up front and can propose
from trial 0.

### MNIST and CIFAR-10 ResNet, 16 dimensions

![MNIST and CIFAR-10 five-seed benchmarks](docs/assets/classification_benchmarks.png)

| method | MNIST cumulative error ↓ | MNIST final error ↓ | CIFAR-10 cumulative error ↓ | CIFAR-10 final error ↓ |
|---|---:|---:|---:|---:|
| Random | 9.174 | 0.648% | 278.920 | 25.072% |
| TPE | 7.166 | 0.580% | 279.936 | 25.596% |
| **GPT-5.5 w/ context** | **5.668** | **0.506%** | **220.994** | **21.322%** |
| GPT-5.5 w/o context | 8.910 | 0.632% | 281.466 | 25.960% |

GPT-5.5 w/ context reduces cumulative best-so-far error by **20.9%** relative to
TPE on MNIST and by **20.8%** relative to Random on CIFAR-10. Without context,
it is 24.3% worse than TPE on MNIST and 0.9% worse than Random on CIFAR-10. The
gap includes both semantic parameter information and earlier access to
agent-guided proposals.

Both [`examples/mnist.py`](examples/mnist.py) and
[`examples/cifar10.py`](examples/cifar10.py) tune learning rate, batch size,
weight decay, label smoothing, three stage widths, three stage depths, and four
dropout controls. MNIST adds translation and rotation; CIFAR-10 uses crop
padding and flip probability.

### Branin and Ackley-5D

Hard-function agents receive **no supplied task context**: only generic
`x1...x5` parameter names, numeric bounds, and trial history. They run in fresh
empty working directories, although a model may still infer a standard
function from its bounds and observations. Every agent uses medium effort for
10 trials over five seeds; Random and TPE are unchanged baselines.

#### Top-tier agents

![No-context top-tier hard-function benchmark](docs/assets/hard_benchmarks_tier.png)

| method | mean best Branin ↓ | mean best Ackley-5D ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| GPT-5.5 | 1.326 | 3.960 |
| **Opus-4.8** | **0.398** | **0.061** |
| Sonnet-5 | 3.850 | 0.143 |
| GLM-5.2 | 3.609 | 15.023 |

The pinned models are `gpt-5.5`, `claude-opus-4-8`, `claude-sonnet-5`, and
`glm-5.2`.
Opus-4.8 reaches the Branin optimum on average and has the strongest five-seed
Ackley mean.

#### Free OpenCode agents

![No-context free-model hard-function benchmark](docs/assets/hard_benchmarks_free.png)

| method | mean best Branin ↓ | mean best Ackley-5D ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| Big-pickle | 4.734 | 15.951 |
| DeepSeek-V4-Flash | 4.410 | **4.608** |
| Nemotron-3-Ultra | 16.051 | 18.459 |
| **MiMo-v2.5** | **3.682** | 15.597 |

These OpenCode-hosted models require no paid model API, making the agent
benchmark practical for students and independent users. The free pool rotates;
this refresh pins `opencode/big-pickle`, `opencode/deepseek-v4-flash-free`,
`opencode/nemotron-3-ultra-free`, and `opencode/mimo-v2.5-free`. DeepSeek V4
Flash has the strongest free-model Ackley mean, while MiMo-v2.5 has the
strongest free-model Branin mean.

Reproduce the distributed runs (the `examples` extra installs Optuna):

```bash
pip install -e ".[examples]"

# Ten classification workers: 2 datasets × 5 seeds. Run no-context first so
# the default verifier can report all four candidates after baselines/context.
python scripts/verify_classification_cumulative_error.py run-no-context
python scripts/verify_classification_cumulative_error.py

# Validate every pinned backend/model first, then run all no-context agents.
python examples/hard_functions.py preflight
python examples/hard_functions.py distributed --trials 10 --seeds 0 1 2 3 4
python examples/hard_functions.py plot
```

## Usage guide

### Sampler effort

| effort | history shown | explicit reasoning | qualitative notes |
|---|---|---|---|
| `low` | last 5 trials | – | – |
| `medium` | last 10 trials | ✓ | ✓ carried across trials |
| `high` | last 20 trials | ✓ | ✓ carried across trials |

Higher effort spends more tokens per trial. For fast objectives, `low` or plain
`RandomSampler()` may be all you need.

### Pruning

```python
study = oa.create_study(
    sampler=oa.AgentSampler(backend="codex"),
    pruner=oa.AgentPruner(backend="codex", level="medium"),  # loose | medium | tight
)

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True,
                             context="learning rate for training an image classifier")
    for epoch in range(20):
        loss = train_one_epoch(lr)
        trial.report(loss, epoch)
        if trial.should_prune():
            raise oa.TrialPruned()
    return loss
```

The pruner agent compares the current learning curve against completed trials
and answers prune/keep; `loose` intervenes only on hopeless runs, `tight`
kills anything underperforming. It never prunes on an agent error.

### Concurrency & distributed studies

Set `max_concurrency` (default `1`) to evaluate several trials at once, and use
a SQLite `storage` file (`.db` / `.sqlite`) as the concurrency-safe shared
history:

```python
study = oa.create_study(
    sampler=oa.AgentSampler(backend="claude"),
    storage="study.db",        # SQLite → safe for many workers; .json stays single-writer
    max_concurrency=8,         # up to 8 objectives run at once
)
study.optimize(objective, n_trials=100)
```

- **Within a process**, `max_concurrency` runs objectives in a thread pool. The
  agent sampling queries are **queued** (serialized) so each proposal sees the
  in-process history; only your `objective` runs in parallel — ideal when it is
  I/O- or subprocess-bound (training a model, hitting an API).
- **Across processes / machines**, point them all at the same SQLite `storage`.
  The database *is* the communication channel: WAL mode lets every worker append
  results and read history without clobbering, and trial numbers stay unique.

Ceilings (deliberate): threads share the GIL, so pure-Python CPU-bound
objectives won't speed up — spread those over processes via shared SQLite
instead. Concurrent workers don't see each other's *in-flight* points, so they
may occasionally probe nearby regions; that is the normal cost of parallel optimization.

### Skill mode (agent reads your code)

The pip package treats your objective as a blackbox. The
[optim-agent skill](SKILL.md) goes further: installed into a
coding-agent session, the agent first *reads your project* to understand each
parameter's role, then drives the same study loop itself via
`study.ask(params)` / `study.tell(trial, value)` — with the study JSON keeping
history across sessions.

```text
$skill-installer install https://github.com/Optim-Agent/optim-agent
```

```python
trial = study.ask({"threshold": 0.72, "budget": 80})
study.tell(trial, evaluate_system(**trial.params))
```

### Offline testing

`AgentSampler(backend="mock")` is a token-free stand-in (hill climbing around
the best point) so you can wire everything up before spending agent calls.

## Troubleshooting

- **`claude` returns 401 inside an agent session** — nested sessions inherit
  `ANTHROPIC_API_KEY`; run with `env -u ANTHROPIC_API_KEY` or from a clean shell.
- **A backend call times out or emits garbage** — the sampler warns and falls
  back to a random point for that trial; the study keeps going.

## Contributing

Contributions are welcome. To develop locally:

```bash
pip install -e ".[examples]"
pytest                     # runs tests/test_optim_agent.py
```

Please open an issue to discuss larger changes before sending a PR. Adding a new
agent backend usually means one small function in [`optim_agent/agent.py`](optim_agent/agent.py).

## Acknowledgements

- [Optuna](https://github.com/optuna/optuna) for popularizing the Study/Trial
  interface, providing the TPE baseline used throughout the examples and
  benchmarks, and setting a high standard for practical optimization tooling.

## License

[MIT](LICENSE)
