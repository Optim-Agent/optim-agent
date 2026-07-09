<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/optim-agent-logo-dark.svg">
    <img alt="optim-agent" src="docs/assets/optim-agent-logo-light.svg" width="500">
  </picture>
</p>

<h1 align="center">optim-agent</h1>

<p align="center">
  <strong>LLM agents as your hyperparameter optimizer.</strong><br>
  A context-aware, agent-driven drop-in for Optuna — for tuning any machine-learning or deep-learning training run.
</p>

<p align="center">
  <a href="https://pypi.org/project/optim-agent/"><img alt="PyPI" src="https://img.shields.io/pypi/v/optim-agent"></a>
  <a href="https://pypi.org/project/optim-agent/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/optim-agent"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/pypi/l/optim-agent"></a>
  <img alt="Zero runtime deps" src="https://img.shields.io/badge/runtime%20deps-0-brightgreen">
</p>

Instead of an evolutionary algorithm or a Bayesian surrogate, optim-agent hands
the *choose-the-next-point* decision to a coding agent (Claude Code, Codex, or
OpenCode) that reasons both **qualitatively** — what a learning rate or a
lookback window *means* — and **quantitatively** — what the trial history
*shows*. No API keys, no extra services: if the agent CLI runs on your machine,
optim-agent can drive it.

## Why optim-agent

- 🧠 **Agent-friendly by design** — the optimizer *is* an LLM agent. Point it at
  `claude`, `codex`, or `opencode`; it reads the trial history and reasons about
  the next configuration like a practitioner would.
- 🎯 **Context-aware** — tell it what each knob *means* (`context="learning rate
  for a CNN"`) and it optimizes with domain priors, not blind point-picking.
- ⚡ **More efficient** — reaches good configs in a handful of trials where
  classical HPO needs a warm-up; the agent pruner kills doomed runs early to
  save compute.
- 🏆 **More effective** — at small budgets (10 trials) top agents hit the optimum
  of standard benchmark functions while TPE and random search don't yet. See
  [Benchmarks](#benchmarks-agents-vs-tpe-and-random-search).
- 🔧 **A general ML/DL training helper** — a drop-in `create_study` /
  `suggest_float` / `optimize` API that wraps *any* training loop, plus a
  [skill](skills/optim-agent/SKILL.md) where the agent reads your code first.
- 📦 **Zero runtime dependencies** — pure stdlib; agents are called through their
  own CLIs. Nothing to host, no keys to manage.

## What you get in practice

- **Small-budget leverage** — useful when each trial is expensive and classical
  surrogates are still data-starved.
- **Semantic tuning** — combines parameter meanings, study context, and trial
  history instead of treating every knob as an anonymous coordinate.
- **Auditable decisions** — records the context, proposals, outcomes, and
  optional agent rationales that matter in high-stakes, risk-managed ML
  workflows.
- **Bounded autonomy** — the agent proposes, optim-agent validates, and your
  objective decides; invalid agent output falls back to safe sampling.
- **Drop-in deployment** — Optuna-style API, JSON/SQLite storage, local
  authenticated CLI backends, and zero runtime dependencies.

**Full documentation:** [docs/index.html](docs/index.html) — served as a
website via GitHub Pages (Settings → Pages → deploy from branch, `main` /`docs`).

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
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True,
                             context="learning rate for training an image classifier")
    batch = trial.suggest_int("batch", 8, 256, log=True,
                              context="mini-batch size; larger is more stable but slower")
    return train_and_validate(lr, batch)          # your code

study = oa.create_study(
    direction="minimize",
    sampler=oa.AgentSampler(
        backend="claude",                          # or "codex" / "opencode"
        effort="high",                             # low | medium | high
        context="a CNN on MNIST",                  # study-wide description (optional)
    ),
    storage="study.json",                          # optional: persist & resume
)
study.optimize(objective, n_trials=20)
print(study.best_value, study.best_params)
```

`context` is optional but powerful: it tells the agent what the parameters
*are*, so it can reason like a practitioner ("loss diverged at lr=0.1 with a
small batch — try 3e-4 and a larger batch") instead of a blind point-picker.
Set it study-wide on `AgentSampler(context=...)`, per-parameter on each
`suggest_*(..., context=...)`, or both — every piece is shown to the agent.

## Benchmarks: agents vs. TPE and random search

Two standard test functions — **Branin** (2D) and **Ackley** (5D) — minimized in
a budget of **10 trials**, mean of **3 seeds**. Agents are told only the input
bounds and the trial history, *never the function name*, so they cannot recall a
known optimum. Baselines: uniform **random search** and **Optuna's TPE** (a
classical Bayesian optimizer). Every agent curve is a real run through the
corresponding CLI.

**Headline agents** — Opus-4.8 and GPT-5.5 reach the optimum of both functions
within the budget, well ahead of TPE and random:

![headline agents](docs/assets/hard_benchmarks_tier.png)

**Free models, no paid API** — models served free by opencode are genuinely
competitive. Big-pickle and DeepSeek-V4 solve Ackley-5D outright and beat random
search on Branin. If you're a student or hobbyist without a paid API key, you can
run optim-agent at **zero model cost**:

![free models](docs/assets/hard_benchmarks_free.png)

Best value reached (mean of 3 seeds, lower is better):

| method | backend | Branin → 0.398 | Ackley-5D → 0 |
|---|---|--:|--:|
| Opus-4.8 | claude | **0.40** | **0.00** |
| GPT-5.5 | codex | **0.40** | **0.00** |
| GLM-5.2 | opencode | 4.26 | **0.00** |
| Big-pickle *(free)* | opencode | 4.26 | **0.00** |
| DeepSeek-V4 *(free)* | opencode | 4.02 | 0.09 |
| Hy3 *(free)* | opencode | 3.75 | 18.72 |
| MiMo-v2.5 *(free)* | opencode | 9.94 | 15.34 |
| TPE (baseline) | optuna | 12.60 | 18.00 |
| Random (baseline) | — | 4.72 | 19.83 |

At a 10-trial budget TPE has too little data to fit a useful surrogate, so it
does not yet beat random — which is exactly the low-sample regime where an
agent's prior knowledge pays off. This is a small-budget demonstration (10
trials, 3 seeds, so still noisy); a multi-seed study with more trials and ML
tasks (MNIST, ARIMA) is in the [paper](paper/README.md).

Reproduce from a clone (the `examples` extra pulls in `optuna` for TPE):

```bash
pip install -e ".[examples]"
for s in 0 1 2; do
  python examples/hard_functions.py run --agent Opus-4.8   --seed $s   # claude
  python examples/hard_functions.py run --agent GPT-5.5    --seed $s   # codex (slow: --timeout 600)
  python examples/hard_functions.py run --agent Big-pickle --seed $s   # free, via opencode
  python examples/hard_functions.py run --agent TPE        --seed $s
  python examples/hard_functions.py run --agent Random     --seed $s
done
python examples/hard_functions.py plot        # writes both figures, averaged over seeds
```

opencode's free roster rotates; check `opencode models | grep -E 'free|pickle'`
and swap model ids in `examples/hard_functions.py` as needed. (Some free entries
are too slow to serve as a sampler and are excluded.)

### MNIST ResNet: NAS-style search space

Full MNIST train/test splits, a small ResNet, **24 trials × 3 seeds**, and 3
epochs per trial. Trials were run in parallel across the machine's **8 A100
GPUs**. Codex used the CLI default model, labeled here as GPT-5.5. TPE is
Optuna's baseline sampler. Lower test error is better.

The search space is defined in [`examples/mnist.py`](examples/mnist.py):
`lr` is a log-uniform float from `1e-5` to `5e-2`; `batch_size` is one of
`64`, `128`, `256`, `512`; `weight_decay` is a log-uniform float from `1e-6`
to `1e-2`; `label_smoothing` is a float from `0.0` to `0.2`; the three
ResNet stages independently tune width, depth, and dropout; `head_dropout` is
a float from `0.0` to `0.8`; `aug_shift` is one of `0`, `1`, `2`, `3`
translation pixels; and `aug_rotate` is one of `0`, `5`, `10` degrees.

The previous GPT-5.5 effort results were generated before the effort ladder was
simplified to `low` / `medium` / `high`. Re-run `examples/mnist.py` before
publishing a fresh three-effort table.

The offline `mock` backend is intentionally excluded from this comparison; it
is only a token-free wiring check, not a real agent call.

### CIFAR-10 ResNet: wider search space

Full CIFAR-10 train/test splits, a small ResNet, **12 trials × 3 seeds**, and
3 epochs per trial. Trials were run in parallel across the machine's **8 A100
GPUs**. Codex used the CLI default model, labeled here as GPT-5.5, with
`model_reasoning_effort` set to `low`, `medium`, and `high`; TPE is Optuna's
baseline sampler. Lower test error is better.

The search space is defined in [`examples/cifar10.py`](examples/cifar10.py):
`lr` is a log-uniform float from `1e-5` to `5e-2`; `batch_size` is one of
`64`, `128`, `256`, `512`; `dropout` is a float from `0.0` to `0.7`; `width`
is one of `16`, `32`, `64`, `96`, `128`, `160` base ResNet channels;
`weight_decay` is a log-uniform float from `1e-6` to `1e-2`; `depth` is one of
`1`, `2`, `3` residual blocks per stage; `label_smoothing` is a float from
`0.0` to `0.2`; `aug_crop` is one of `0`, `2`, `4`, `6` random-crop padding
pixels; and `aug_flip` is one of `0.0`, `0.5` horizontal flip probability.

![CIFAR-10 GPT-5.5 effort sweep](docs/assets/cifar10_benchmarks.png)

Best test error reached:

| method | backend/effort | seed 0 | seed 1 | seed 2 | mean best test error |
|---|---|--:|--:|--:|--:|
| Random | baseline | 28.16% | 25.45% | 30.56% | 28.06% |
| TPE | optuna / TPE | 30.32% | 20.62% | 33.49% | 28.14% |
| GPT-5.5-low | codex / low | 27.71% | 27.70% | 30.08% | 28.50% |
| GPT-5.5-medium | codex / medium | 28.35% | 23.94% | 29.00% | 27.10% |
| GPT-5.5-xhigh (legacy) | codex / xhigh | 26.90% | 26.49% | 26.11% | 26.50% |

The offline `mock` backend is intentionally excluded from this comparison; it
is only a token-free wiring check, not a real agent call.

## Usage guide

### Sampler effort

| effort | history shown | explicit reasoning | qualitative notes |
|---|---|---|---|
| `low` | last 5 trials | – | – |
| `medium` | last 10 trials | ✓ | – |
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
may occasionally probe nearby regions; that is the normal cost of parallel HPO.

### Skill mode (agent reads your code)

The pip package treats your objective as a blackbox. The
[optim-agent skill](skills/optim-agent/SKILL.md) goes further: installed into a
coding-agent session, the agent first *reads your project* to understand each
hyperparameter's role, then drives the same study loop itself via
`study.ask(params)` / `study.tell(trial, value)` — with the study JSON keeping
history across sessions.

```python
trial = study.ask({"lr": 3e-4, "batch": 64})   # the session agent picks the point
study.tell(trial, run_training(**trial.params))
```

### Offline testing

`AgentSampler(backend="mock")` is a token-free stand-in (hill climbing around
the best point) so you can wire everything up before spending agent calls.

## Ablations

Both ablations fix the model (**GPT-5.5 via codex**) and vary one knob, on the
same Branin/Ackley functions with Random and TPE for reference. Effort is
forwarded to codex's `model_reasoning_effort`, so higher effort genuinely makes
the model deliberate harder (`python examples/ablations.py plot`).

### Sampler effort

![sampler effort ablation](docs/assets/abl_effort.png)

GPT-5.5 at three efforts (`low`/`medium`/`high`), best value vs trial, mean of 3 seeds.
**Every effort reaches the optimum of both functions**, far ahead of Random and
TPE. Effort barely separates — but now for a telling reason: these benchmarks are
*too easy for GPT-5.5*. It saturates at every effort, so there is no headroom for
more reasoning to help. The one visible effect is **speed of convergence on the
harder function**: on Ackley-5D the higher efforts reach 0 by trial ~4 while
`low` needs until trial ~8; on the easy Branin all efforts collapse to the
optimum together by trial ~4. Effort buys faster convergence when the problem is
hard enough to reward it; on trivial objectives `low` is all you need. The
[MNIST sweep](#mnist-resnet-nas-style-search-space) should be re-run before
drawing conclusions under the simplified effort ladder.

### Pruner tightness

![pruner tightness ablation](docs/assets/abl_prune.png)

Branin and Ackley are scalar, so there is no learning curve for a pruner to
watch. To exercise pruning we attach a **synthetic** noisy loss curve (four
steps descending toward `f(x)`, with occasional slow-starters) to each
evaluation; the x-axis is **compute (reported steps)**, mean of 2 seeds. A
pruner's payoff is compute saved, so this plots best value vs steps.

- **`loose` / `medium` are a clear win**: they reach the optimum at *less*
  compute than no pruning. On Ackley-5D `loose` hits 0 by ~16 steps and `medium`
  by ~23, where `none` needs the full 40 — pruning the doomed trials early frees
  budget without giving up quality.
- **`tight` over-prunes**: it stops at ~14–20 steps but abandons good trials
  before they prove themselves, stalling at Branin ≈6.9 and Ackley ≈20.

Lesson: moderate pruning saves real compute at near-optimal quality; aggressive
pruning trades away quality it shouldn't. Prefer `loose`/`medium` unless each
evaluation is so expensive that cutting late-blooming winners is worth it.

```bash
# uses GPT-5.5 (codex); Random/TPE curves are reused from the benchmark above
for s in 0 1 2; do
  for e in low medium high; do python examples/ablations.py effort --variant $e --seeds $s; done
done
for s in 0 1; do
  for p in loose medium tight; do python examples/ablations.py prune --variant $p --seeds $s; done
done
python examples/ablations.py plot
```

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

## Paper

An arXiv paper with extended experiments (MNIST classification, ARIMA
time-series fitting, baseline and ablation studies) is in preparation under
[`paper/`](paper/README.md).

## License

[MIT](LICENSE)
