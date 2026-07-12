# No-Context Hard-Function Agent Rosters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish five-seed, ten-trial no-context Branin/Ackley comparisons for four top-tier agents, four free OpenCode agents, Random, and TPE.

**Architecture:** Keep `examples/hard_functions.py` as the benchmark authority. Expand its pinned pool, factor sampler/provenance validation into testable helpers, add a model-preflight command, and make plot loading reject any incompatible artifact before generating separate tier and free figures. Update the public README and static documentation only after validated measurements exist.

**Tech Stack:** Python 3, optim-agent, Codex CLI, Claude CLI, OpenCode CLI, Optuna, matplotlib, pytest.

---

### Task 1: Lock The Roster And No-Context Contract With Failing Tests

**Files:**
- Modify: `tests/test_optim_agent.py:691`
- Test: `tests/test_optim_agent.py`

- [ ] **Step 1: Replace the old four-candidate assertions**

Update `test_hard_functions_distributed_contract` to assert the exact ten-entry pool:

```python
    expected = {
        "Random": (None, None, "both"),
        "TPE": ("tpe", None, "both"),
        "GPT-5.5": ("codex", "gpt-5.5", "tier"),
        "Opus-4.8": ("claude", "claude-opus-4-8", "tier"),
        "Sonnet-5": ("claude", "claude-sonnet-5", "tier"),
        "GLM-5.2": ("opencode", "cmkey/glm-5.2", "tier"),
        "Big-pickle": ("opencode", "opencode/big-pickle", "free"),
        "DeepSeek-V4-Flash": (
            "opencode", "opencode/deepseek-v4-flash-free", "free"
        ),
        "Nemotron-3-Ultra": (
            "opencode", "opencode/nemotron-3-ultra-free", "free"
        ),
        "MiMo-v2.5": ("opencode", "opencode/mimo-v2.5-free", "free"),
    }
    assert {
        label: (preset["backend"], preset["model"], preset["group"])
        for label, preset in hard.POOL.items()
    } == expected
```

- [ ] **Step 2: Assert every agent sampler is medium-effort and no-context**

Add these assertions to the same test:

```python
    for label, preset in hard.POOL.items():
        if preset["backend"] in (None, "tpe"):
            continue
        sampler = hard._make_sampler(preset, seed=3, timeout=17)
        assert sampler.backend == preset["backend"]
        assert sampler.model == preset["model"]
        assert sampler.effort == "medium"
        assert sampler.context is None
        assert sampler.n_init == 3
        assert sampler.timeout == 17
```

- [ ] **Step 3: Add preflight call-contract coverage**

Add a new test that records calls without invoking external CLIs:

```python
def test_hard_functions_preflight_contract():
    from examples import hard_functions as hard

    calls = []
    old_call = hard.agent_api.call_agent
    hard.agent_api.call_agent = lambda backend, model, prompt, timeout, effort: (
        calls.append((backend, model, timeout, effort)) or '{"ok": true}'
    )
    try:
        hard.preflight(timeout=23)
    finally:
        hard.agent_api.call_agent = old_call

    expected = [
        (preset["backend"], preset["model"], 23, "medium")
        for preset in hard.POOL.values()
        if preset["backend"] not in (None, "tpe")
    ]
    assert calls == expected
```

Also add `test_hard_functions_preflight_contract` to the direct-execution test list at the bottom of the file.

- [ ] **Step 4: Run the targeted tests and verify RED**

Run:

```bash
python -m pytest -q tests/test_optim_agent.py \
  -k 'hard_functions_distributed_contract or hard_functions_preflight_contract'
```

Expected: failures because the expanded pool, `_make_sampler`, `agent_api`, and `preflight` do not yet exist.

### Task 2: Implement The Pinned No-Context Benchmark Matrix

**Files:**
- Modify: `examples/hard_functions.py:1-235`
- Test: `tests/test_optim_agent.py`

- [ ] **Step 1: Replace the pool and shared constants**

Add `from optim_agent import agent as agent_api`, define `AGENT_EFFORT = "medium"`, and replace `POOL` with the exact roster from Task 1. Preserve distinct colors and line styles, and keep Random/TPE in group `both`.

The agent entries must use these exact model strings:

```python
"GPT-5.5": "gpt-5.5"
"Opus-4.8": "claude-opus-4-8"
"Sonnet-5": "claude-sonnet-5"
"GLM-5.2": "cmkey/glm-5.2"
"Big-pickle": "opencode/big-pickle"
"DeepSeek-V4-Flash": "opencode/deepseek-v4-flash-free"
"Nemotron-3-Ultra": "opencode/nemotron-3-ultra-free"
"MiMo-v2.5": "opencode/mimo-v2.5-free"
```

- [ ] **Step 2: Factor sampler construction and remove contextual hard-function prompting**

Delete `_context`. Add:

```python
def _is_agent(preset):
    return preset["backend"] not in (None, "tpe")


def _make_sampler(preset, seed, timeout):
    if preset["backend"] is None:
        return oa.RandomSampler()
    if preset["backend"] == "tpe":
        raise ValueError("TPE uses Optuna directly")
    return oa.AgentSampler(
        backend=preset["backend"],
        model=preset["model"],
        effort=AGENT_EFFORT,
        context=None,
        n_init=3,
        timeout=timeout,
        seed=seed,
        agent_cwd=agent_cwd,
    )
```

Use `_make_sampler` inside `_agent_curve`. Create a fresh empty
`TemporaryDirectory` for each function/seed curve and pass it as `agent_cwd`.
No agent path may pass `_context(spec)` or any other study description, and no
child CLI may inherit the repository working directory.

- [ ] **Step 3: Add exact artifact provenance**

Build each result with:

```python
    out = {
        "label": label,
        "backend": preset["backend"],
        "model": preset["model"],
        "effort": AGENT_EFFORT if _is_agent(preset) else None,
        "no_context": True if _is_agent(preset) else None,
        "seed": seed,
        "trials": trials,
        "functions": {},
    }
```

- [ ] **Step 4: Add strict run validation**

Add `_validate_run(label, run)` and call it from `_load_plot_runs`. It must reject data unless:

```python
run["label"] == label
run["backend"] == POOL[label]["backend"]
run["model"] == POOL[label]["model"]
run["seed"] in range(5)
run["trials"] == 10
set(run["functions"]) == set(FUNCTIONS)
len(values) == len(params) == 10 for each function
run["effort"] == "medium" and run["no_context"] is True for agents
run["effort"] is None and run["no_context"] is None for baselines
```

`_load_plot_runs` must require exactly `set(POOL)`, exactly five files per label, and seeds `0..4`.

- [ ] **Step 5: Restore separate group plots**

Iterate labels in `POOL` order, include group-specific agents plus `both` baselines, and generate:

```python
_plot_group("tier", by_label, "hard_benchmarks_tier.png")
_plot_group("free", by_label, "hard_benchmarks_free.png")
```

Use titles `Top-tier agents, medium effort, no context` and `Free OpenCode agents, medium effort, no context`, followed by the shared five-seed/baseline description.

- [ ] **Step 6: Implement the model preflight subcommand**

Add:

```python
def preflight(timeout):
    prompt = 'Reply with ONLY this JSON object: {"ok": true}'
    for label, preset in POOL.items():
        if not _is_agent(preset):
            continue
        reply = agent_api.call_agent(
            preset["backend"], preset["model"], prompt, timeout,
            effort=AGENT_EFFORT,
        )
        data = agent_api.extract_json(reply)
        if not data or data.get("ok") is not True:
            raise RuntimeError(f"preflight failed for {label}: invalid JSON reply")
        print(f"preflight ok: {label} ({preset['backend']} {preset['model']})")
```

Register `preflight` with `--timeout`, alongside `run`, `distributed`, `plot`, and `selfcheck`.

- [ ] **Step 7: Run targeted tests and verify GREEN**

Run the command from Task 1 Step 4.

Expected: both selected tests pass.

OpenCode-backed labels must use one seed worker because concurrent OpenCode
processes share a local database. All other labels use `len(seeds)` workers.

- [ ] **Step 8: Run all non-PyTorch tests**

Run:

```bash
python -m pytest -q \
  -k 'not test_mnist_helper_curves_and_labels and not test_cifar10_helper_curves_and_labels'
```

Expected: all selected tests pass. The two deselected classification tests require optional PyTorch, which is absent from this environment.

### Task 3: Verify Every External Model Before Publishing

**Files:**
- No repository files changed.
- Consumes: the `preflight` command from Task 2.

- [ ] **Step 1: Confirm OpenCode still advertises the pinned models**

Run:

```bash
opencode models
```

Expected: `cmkey/glm-5.2` and all four configured free OpenCode IDs are present.

- [ ] **Step 2: Run all eight structured-output smoke tests**

Run:

```bash
python -u examples/hard_functions.py preflight --timeout 600
```

Expected: eight `preflight ok:` lines, including GPT-5.5 resolving through Codex with explicit model `gpt-5.5`. If any model fails, stop before deleting artifacts and report the exact failure.

### Task 4: Generate And Validate All Benchmark Artifacts

**Files:**
- Replace/Create: `docs/assets/hard_curves_*.json`
- Replace/Create: `docs/assets/hard_benchmarks_tier.png`
- Replace/Create: `docs/assets/hard_benchmarks_free.png`

- [ ] **Step 1: Remove all previous hard-function publication artifacts**

Only after Task 3 passes, run:

```bash
rm -f docs/assets/hard_curves_*.json \
      docs/assets/hard_benchmarks_tier.png \
      docs/assets/hard_benchmarks_free.png
```

- [ ] **Step 2: Run the ten-method, five-seed matrix**

Run in the foreground with unbuffered output:

```bash
python -u examples/hard_functions.py distributed \
  --trials 10 --seeds 0 1 2 3 4 --timeout 600
```

Expected: ten sequential method batches, each with five concurrent seeds, and no `agent call failed`, `falling back`, `unparseable`, or timeout warning.

- [ ] **Step 3: Validate the 50 JSON files through the publication loader**

Run:

```bash
python - <<'PY'
from examples import hard_functions as hard
runs = hard._load_plot_runs()
assert set(runs) == set(hard.POOL)
assert sum(map(len, runs.values())) == 50
print("validated 50 no-context hard-function curves")
PY
```

Expected: `validated 50 no-context hard-function curves`.

- [ ] **Step 4: Regenerate both plots**

Run:

```bash
python examples/hard_functions.py plot
```

Expected: both tier and free PNG paths are printed.

### Task 5: Publish Data-Derived Documentation

**Files:**
- Modify: `README.md:163-183`
- Modify: `docs/index.html:443-455`

- [ ] **Step 1: Derive final means from the validated files**

Run:

```bash
python - <<'PY'
import json, statistics
from pathlib import Path
from examples import hard_functions as hard
hard._load_plot_runs()
for group in ("tier", "free"):
    print(group)
    for label, preset in hard.POOL.items():
        if preset["group"] not in (group, "both"):
            continue
        runs = [json.loads(p.read_text()) for p in sorted(
            Path("docs/assets").glob(f"hard_curves_{label}_s*.json")
        )]
        means = [statistics.mean(min(r["functions"][name]["values"]) for r in runs)
                 for name in hard.FUNCTIONS]
        print(label, *(f"{value:.3f}" for value in means))
PY
```

Expected: six exact data rows for each group: four agents plus Random and TPE.

- [ ] **Step 2: Replace the README hard-function section**

State explicitly that all eight agents use medium effort and no task context. Add one table and plot per roster, list the exact backend/model IDs, explain that OpenCode's free pool rotates, and describe the free roster as requiring no paid model API. Populate numeric cells only from Step 1 output.

- [ ] **Step 3: Mirror the same content in the static documentation page**

Update the hard-function section in `docs/index.html` with the same two images, rosters, no-context statement, rotating-pool note, and data-derived metrics.

- [ ] **Step 4: Check documentation consistency**

Run:

```bash
rg -n "hard_benchmarks_(tier|free)|no context|gpt-5.5|free pool|paid" \
  README.md docs/index.html
```

Expected: both documents reference both plots, explicit GPT-5.5 pinning, no-context semantics, and the free-model value proposition.

### Task 6: Final Verification And Visual Review

**Files:**
- Verify all files changed by Tasks 1-5.

- [ ] **Step 1: Run numerical and automated checks**

Run:

```bash
python examples/hard_functions.py selfcheck
python -m pytest -q \
  -k 'not test_mnist_helper_curves_and_labels and not test_cifar10_helper_curves_and_labels'
git diff --check
```

Expected: self-check passes, all selected tests pass, and `git diff --check` emits no output.

- [ ] **Step 2: Re-run strict publication validation**

Run Task 4 Step 3 again after documentation edits.

Expected: 50 validated artifacts.

- [ ] **Step 3: Inspect both PNGs**

Open `docs/assets/hard_benchmarks_tier.png` and
`docs/assets/hard_benchmarks_free.png`. Confirm each has six visible series,
integer trial ticks, readable axes, no clipping, and a legend matching its four
agents plus Random and TPE.

- [ ] **Step 4: Audit final scope**

Run:

```bash
git status --short
git diff --stat
```

Expected: implementation/tests/docs plus 50 hard-function JSON files and two
plots. `graphify-out/` may remain untracked as the requested repository graph;
no classification or paper artifacts should change.
