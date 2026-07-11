# Five-Seed Benchmark Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish five-seed MNIST, CIFAR-10, and hard-functions comparisons of Random, TPE, GPT-5.5 medium, and GPT-5.5 medium without context.

**Architecture:** Extend the existing classification verifier to schedule the missing no-context studies with its current five-seed process pool pattern. Keep hard-functions orchestration local to its example with one thread per seed and preserve the existing per-seed JSON format and plotting path. Publish only validated JSON-derived metrics and four-candidate plots.

**Tech Stack:** Python standard library, optim-agent, Optuna, PyTorch/torchvision, matplotlib, NumPy, Codex CLI.

## Global Constraints

- Use seeds `0, 1, 2, 3, 4` and 10 trials per seed.
- Explicitly select Codex model `gpt-5.5` and reasoning effort `medium` for every GPT run.
- Classification reward is `sum(best_test_error_so_far[i] for i in 1..N)`; lower is better.
- All plots contain only Random, TPE, GPT-5.5 medium, and GPT-5.5 medium without context.
- Reuse the fresh Random, TPE, and context-enabled classification artifacts under `autoresearch-results/classification-stagewise16-v2-n10-s5`.
- Run failed workers must propagate a nonzero exit and plots are generated only from five validated seeds.
- Do not add dependencies or a shared orchestration framework.

---

### Task 1: Classification No-Context Runner Contract

**Files:**
- Modify: `tests/test_optim_agent.py:511-580`
- Modify: `scripts/verify_classification_reward.py:16-184`
- Modify: `examples/mnist.py:491-531`
- Modify: `examples/cifar10.py:487-529`

**Interfaces:**
- Consumes: existing `_run_pair(method: str, root: Path)` and dataset `run(...)` functions.
- Produces: `GPT_NO_CONTEXT: Path`, `run-no-context` CLI command, and curve metadata keys `model` and `effort`.

- [ ] **Step 1: Add failing no-context contract assertions**

Extend `test_verify_classification_reward_contract()` with:

```python
assert verify.LABELS["codex-no-context"] == "GPT-5.5-medium-no-context"
assert verify.GPT_NO_CONTEXT.name == "gpt-no-context"

seen = {}
class FakeModule:
    _sampler = staticmethod(lambda method, *args: (
        seen.update(sampler_method=method) or SimpleNamespace(anchor_proposals=[])
    ))
    run = staticmethod(lambda method, seeds, *args: seen.update(method=method, seeds=seeds))

old_dataset_module = verify._dataset_module
verify._dataset_module = lambda dataset: FakeModule
try:
    verify._worker(SimpleNamespace(dataset="mnist", method="codex-no-context", seed=3,
                                   assets="/tmp/assets", storage="/tmp/storage", gpus=[0]))
finally:
    verify._dataset_module = old_dataset_module
assert seen == {"sampler_method": "codex-no-context",
                "method": "codex-no-context", "seeds": [3]}
```

- [ ] **Step 2: Run the focused test and confirm the contract is absent**

Run:

```bash
python tests/test_optim_agent.py
```

Expected: failure because `GPT_NO_CONTEXT` and the new label do not exist or because `_worker()` still builds a `codex` sampler.

- [ ] **Step 3: Extend the verifier minimally**

Add the result root and label:

```python
GPT_NO_CONTEXT = RUN_ROOT / "gpt-no-context"
LABELS = {
    "Random": "Random",
    "TPE": "TPE",
    "codex": "GPT-5.5-medium",
    "codex-no-context": "GPT-5.5-medium-no-context",
}
```

Change `_worker()` to preserve the requested method:

```python
sampler = module._sampler(args.method, args.seed, EFFORT, TIMEOUT, MODEL)
```

Add a CLI subcommand and handler without changing the existing default verifier path:

```python
sub.add_parser("run-no-context")
# after parsing and before the default full verification path
if args.command == "run-no-context":
    shutil.rmtree(GPT_NO_CONTEXT, ignore_errors=True)
    _run_pair("codex-no-context", GPT_NO_CONTEXT)
    return
```

Extend `_metrics()` without changing the existing baseline-ratio success metric:

```python
no_context_reward, no_context_seeds = _reward(
    GPT_NO_CONTEXT, dataset, LABELS["codex-no-context"]
)
metrics[f"{prefix}_gpt_no_context_reward"] = no_context_reward
metrics.update({f"{prefix}_gpt_no_context_s{i}": value
                for i, value in enumerate(no_context_seeds)})
```

Add explicit run metadata to both classification example output dictionaries:

```python
"model": model if METHODS[method]["backend"] == "codex" else None,
```

- [ ] **Step 4: Run the contract and example self-checks**

Run:

```bash
python tests/test_optim_agent.py
python examples/mnist.py selfcheck
python examples/cifar10.py selfcheck
```

Expected: all commands exit zero.

- [ ] **Step 5: Commit the classification runner**

```bash
git add tests/test_optim_agent.py scripts/verify_classification_reward.py examples/mnist.py examples/cifar10.py
git commit -m "feat: run no-context classification seeds in parallel"
```

### Task 2: Four-Candidate Hard-Functions Runner

**Files:**
- Modify: `tests/test_optim_agent.py:665-682`
- Modify: `examples/hard_functions.py:1-201`

**Interfaces:**
- Consumes: `run(label: str, trials: int, seed: int, timeout: float)`.
- Produces: `run_distributed(labels: list[str], trials: int, seeds: list[int], timeout: float)` and the `distributed` CLI command.

- [ ] **Step 1: Add a failing distributed scheduling test**

Add this test and include it in the manual test list at the bottom of `tests/test_optim_agent.py`:

```python
def test_hard_functions_distributed_contract():
    from examples import hard_functions as hard
    import threading

    assert tuple(hard.POOL) == (
        "Random", "TPE", "GPT-5.5-medium", "GPT-5.5-medium-no-context",
    )
    assert hard.POOL["GPT-5.5-medium"]["model"] == "gpt-5.5"
    assert hard.POOL["GPT-5.5-medium-no-context"]["no_context"] is True

    barrier = threading.Barrier(5, timeout=5)
    calls = []
    old_run = hard.run
    hard.run = lambda label, trials, seed, timeout: (
        calls.append((label, trials, seed, timeout)), barrier.wait()
    )
    try:
        hard.run_distributed(["Random"], 10, [0, 1, 2, 3, 4], 600)
    finally:
        hard.run = old_run
    assert {call[2] for call in calls} == {0, 1, 2, 3, 4}
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
python tests/test_optim_agent.py
```

Expected: failure because the four-candidate pool and `run_distributed()` do not exist.

- [ ] **Step 3: Replace the legacy pool and pin GPT configuration**

Use only:

```python
POOL = {
    "Random": dict(backend=None, model=None, group="both", color="#9ca3af", style="solid"),
    "TPE": dict(backend="tpe", model=None, group="both", color="#111827", style=(0, (2, 2))),
    "GPT-5.5-medium": dict(backend="codex", model="gpt-5.5", group="tier",
                           color="#10a37f", style=(0, (4, 2))),
    "GPT-5.5-medium-no-context": dict(backend="codex", model="gpt-5.5",
                                      no_context=True, group="tier",
                                      color="#10a37f", style=(0, (1, 1.5))),
}
```

Configure the agent in `_agent_curve()` with:

```python
sampler = oa.AgentSampler(
    backend=preset["backend"], model=preset["model"], effort="medium",
    context=None if preset.get("no_context") else _context(spec),
    n_init=3, timeout=timeout, seed=seed,
)
```

Write auditable metadata in `run()`:

```python
out = {
    "label": label,
    "seed": seed,
    "trials": trials,
    "model": preset["model"],
    "effort": "medium" if preset["backend"] == "codex" else None,
    "functions": {},
}
```

- [ ] **Step 4: Add five-seed parallel execution**

Import `ThreadPoolExecutor` and add:

```python
def run_distributed(labels, trials, seeds, timeout):
    for label in labels:
        with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
            futures = [pool.submit(run, label, trials, seed, timeout) for seed in seeds]
            for future in futures:
                future.result()
```

Add the parser:

```python
p_dist = sub.add_parser("distributed")
p_dist.add_argument("--agents", nargs="+", choices=list(POOL), default=list(POOL))
p_dist.add_argument("--trials", type=int, default=10)
p_dist.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
p_dist.add_argument("--timeout", type=float, default=600)
```

Dispatch it with:

```python
"distributed": lambda: run_distributed(args.agents, args.trials, args.seeds, args.timeout),
```

Keep `_plot_group()` and generate only `hard_benchmarks_tier.png`; remove the free-pool call because no free candidates remain.

- [ ] **Step 5: Run the contract and self-check**

Run:

```bash
python tests/test_optim_agent.py
python examples/hard_functions.py selfcheck
```

Expected: both commands exit zero and the scheduling test observes all five seeds concurrently.

- [ ] **Step 6: Commit the hard-functions runner**

```bash
git add tests/test_optim_agent.py examples/hard_functions.py
git commit -m "feat: distribute four-candidate hard benchmarks"
```

### Task 3: Generate and Publish Classification Artifacts

**Files:**
- Create: `autoresearch-results/classification-stagewise16-v2-n10-s5/gpt-no-context/**`
- Replace/Create: `docs/assets/mnist_curves_*.json`
- Create: `docs/assets/cifar10_curves_*.json`
- Replace: `docs/assets/mnist_benchmarks.png`
- Create: `docs/assets/cifar10_benchmarks.png`

**Interfaces:**
- Consumes: validated baseline/context curves plus Task 1's `run-no-context` output.
- Produces: 20 JSON curves and two four-candidate publication plots.

- [ ] **Step 1: Launch the missing no-context runs in the foreground**

```bash
python scripts/verify_classification_reward.py run-no-context
```

Expected: ten worker processes complete, one for each MNIST/CIFAR-10 and seed combination, with all Codex calls using `gpt-5.5` at medium effort.

- [ ] **Step 2: Validate the new curves before publishing**

```bash
python - <<'PY'
from scripts import verify_classification_reward as v
for dataset in v.GPU_SPLITS:
    label = v.LABELS["codex-no-context"]
    assert v._complete(v.GPT_NO_CONTEXT, dataset, label)
    reward, seeds = v._reward(v.GPT_NO_CONTEXT, dataset, label)
    assert len(seeds) == 5
    print(dataset, label, reward, seeds)
PY
```

Expected: two lines, each containing five numeric seed rewards.

- [ ] **Step 3: Replace stale classification curves with the four approved candidates**

Remove existing published MNIST/CIFAR curve JSON files, then copy the five seeds for Random and TPE from `baselines`, GPT-5.5 medium from `gpt-current`, and GPT-5.5 medium without context from `gpt-no-context` into `docs/assets`.

```bash
rm -f docs/assets/mnist_curves_*.json docs/assets/cifar10_curves_*.json
cp autoresearch-results/classification-stagewise16-v2-n10-s5/baselines/mnist/*.json docs/assets/
cp autoresearch-results/classification-stagewise16-v2-n10-s5/baselines/cifar10/*.json docs/assets/
cp autoresearch-results/classification-stagewise16-v2-n10-s5/gpt-current/mnist/*.json docs/assets/
cp autoresearch-results/classification-stagewise16-v2-n10-s5/gpt-current/cifar10/*.json docs/assets/
cp autoresearch-results/classification-stagewise16-v2-n10-s5/gpt-no-context/mnist/*.json docs/assets/
cp autoresearch-results/classification-stagewise16-v2-n10-s5/gpt-no-context/cifar10/*.json docs/assets/
```

The reused context-enabled runs were launched by the verifier with its explicit
`MODEL = "gpt-5.5"` constant before curve files carried a model field. Record
that provenance in the published copies with structured JSON parsing:

```bash
python - <<'PY'
import json
from pathlib import Path
for dataset in ("mnist", "cifar10"):
    for path in Path("docs/assets").glob(f"{dataset}_curves_GPT-5.5-medium_s*.json"):
        data = json.loads(path.read_text())
        data["model"] = "gpt-5.5"
        path.write_text(json.dumps(data, indent=1))
PY
```

- [ ] **Step 4: Generate the plots**

```bash
python examples/mnist.py plot
python examples/cifar10.py plot
```

Expected: both plot commands report their PNG path and each legend contains four entries marked `n=5`.

- [ ] **Step 5: Commit classification artifacts**

```bash
git add -A docs/assets
git commit -m "results: publish five-seed classification comparison"
```

The authoritative autoresearch directory is excluded from Git by repository
configuration and remains available locally for audit and resume.

### Task 4: Generate and Publish Hard-Functions Artifacts

**Files:**
- Replace/Create: `docs/assets/hard_curves_*.json`
- Replace: `docs/assets/hard_benchmarks_tier.png`
- Delete: `docs/assets/hard_benchmarks_free.png`

**Interfaces:**
- Consumes: Task 2's `distributed` command.
- Produces: 20 hard-functions JSON files and one four-candidate plot.

- [ ] **Step 1: Remove legacy hard-function publication artifacts**

```bash
rm -f docs/assets/hard_curves_*.json docs/assets/hard_benchmarks_free.png
```

- [ ] **Step 2: Run all candidates with five parallel seeds**

```bash
python examples/hard_functions.py distributed --trials 10 --seeds 0 1 2 3 4 --timeout 600
```

Expected: all four candidate batches complete; each batch runs five seeds concurrently and produces five JSON files.

- [ ] **Step 3: Validate metadata and shape**

```bash
python - <<'PY'
import json
from pathlib import Path
labels = ("Random", "TPE", "GPT-5.5-medium", "GPT-5.5-medium-no-context")
root = Path("docs/assets")
for label in labels:
    paths = sorted(root.glob(f"hard_curves_{label}_s*.json"))
    assert len(paths) == 5, (label, len(paths))
    for path in paths:
        data = json.loads(path.read_text())
        assert data["trials"] == 10
        assert all(len(result["values"]) == 10 for result in data["functions"].values())
        if label.startswith("GPT"):
            assert data["model"] == "gpt-5.5"
            assert data["effort"] == "medium"
print("validated 20 hard-function curves")
PY
```

Expected: `validated 20 hard-function curves`.

- [ ] **Step 4: Plot and commit**

```bash
python examples/hard_functions.py plot
git add -A docs/assets
git commit -m "results: publish five-seed hard-function comparison"
```

Expected: `hard_benchmarks_tier.png` is regenerated and the free-pool plot is deleted.

### Task 5: Refresh README and Documentation Page

**Files:**
- Modify: `README.md:107-218`
- Modify: `README.md:256-282`
- Modify: `docs/index.html:400-437`
- Modify: `docs/index.html:534-536`

**Interfaces:**
- Consumes: Tasks 3 and 4 JSON and PNG artifacts.
- Produces: matching benchmark descriptions and reproducible commands in Markdown and HTML.

- [ ] **Step 1: Calculate all publication metrics from JSON**

Run this read-only summary; do not transcribe values from training logs:

```bash
python - <<'PY'
import json
import statistics
from pathlib import Path

root = Path("docs/assets")
labels = ("Random", "TPE", "GPT-5.5-medium", "GPT-5.5-medium-no-context")

for dataset in ("mnist", "cifar10"):
    for label in labels:
        rewards, finals = [], []
        for path in sorted(root.glob(f"{dataset}_curves_{label}_s*.json")):
            data = json.loads(path.read_text())
            best, curve = float("inf"), []
            for record in data["records"]:
                best = min(best, float(record["test_error"]))
                curve.append(best)
            rewards.append(sum(curve))
            finals.append(curve[-1])
        print(dataset, label, "reward", statistics.mean(rewards),
              "final", statistics.mean(finals))

for label in labels:
    runs = [json.loads(path.read_text())
            for path in sorted(root.glob(f"hard_curves_{label}_s*.json"))]
    for function in ("branin", "ackley5"):
        finals = [min(run["functions"][function]["values"]) for run in runs]
        print("hard", function, label, "final", statistics.mean(finals))
PY
```

Expected: one machine-derived summary for all three benchmark families and four labels.

- [ ] **Step 2: Replace the README benchmark section**

Document:

- 10 trials, five seeds, and the four-candidate restriction.
- Explicit `gpt-5.5` and medium effort.
- The classification reward definition and lower-is-better interpretation.
- The 16-dimensional MNIST and CIFAR-10 spaces.
- Both classification plots and the hard-functions plot.
- JSON-derived reward and final-incumbent metrics.
- `python scripts/verify_classification_reward.py run-no-context` and `python examples/hard_functions.py distributed ...` reproduction commands.

Remove legacy free-model, effort-sweep, three-seed, default-model, and old CIFAR search-space claims from the benchmark section. Preserve installation, API, and unrelated usage text. After editing, run `git diff --numstat README.md`; if changed lines exceed 30 percent of 385 lines, rewrite README coherently while retaining the same non-benchmark information.

- [ ] **Step 3: Mirror the benchmark facts in the docs page**

Replace the old three-seed hard-functions-only block with MNIST, CIFAR-10, and hard-functions subsections using the same JSON-derived numbers and images. Update the FAQ claim that parallel trials are unsupported so it reflects the existing `max_concurrency` behavior and independent-study distributed commands.

- [ ] **Step 4: Check documentation references and stale claims**

```bash
rg -n "3 seeds|three seeds|24 trials|12 trials|CLI default model|Opus-4.8|Big-pickle|hard_benchmarks_free|Parallel trials.*Not supported" README.md docs/index.html
```

Expected: no matches in the refreshed benchmark/FAQ content.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md docs/index.html
git commit -m "docs: describe five-seed four-candidate benchmarks"
```

### Task 6: Full Verification and Visual Inspection

**Files:**
- Verify only; fix the smallest owning file if a check fails.

**Interfaces:**
- Consumes: all earlier task outputs.
- Produces: a clean worktree with tested code, complete artifacts, and legible plots.

- [ ] **Step 1: Run the full test and self-check suite**

```bash
python tests/test_optim_agent.py
python examples/mnist.py selfcheck
python examples/cifar10.py selfcheck
python examples/hard_functions.py selfcheck
```

Expected: every command exits zero; the test script ends with `all checks passed`.

- [ ] **Step 2: Verify classification artifact counts and metadata**

```bash
python - <<'PY'
import json
from pathlib import Path
for dataset in ("mnist", "cifar10"):
    paths = sorted(Path("docs/assets").glob(f"{dataset}_curves_*_s*.json"))
    assert len(paths) == 20, (dataset, len(paths))
    labels = {}
    for path in paths:
        data = json.loads(path.read_text())
        labels.setdefault(data["label"], []).append(data)
        assert data["trials"] == 10 and len(data["records"]) == 10
        if data["label"].startswith("GPT"):
            assert data["model"] == "gpt-5.5"
            assert data["effort"] == "medium"
    assert set(labels) == {"Random", "TPE", "GPT-5.5-medium", "GPT-5.5-medium-no-context"}
    assert all(len(runs) == 5 for runs in labels.values())
print("validated 40 classification curves")
PY
```

Expected: `validated 40 classification curves`.

- [ ] **Step 3: Inspect generated images**

Open `docs/assets/mnist_benchmarks.png`, `docs/assets/cifar10_benchmarks.png`, and `docs/assets/hard_benchmarks_tier.png`. Confirm lines and axes are visible, legends contain exactly the four approved candidates, classification entries say `n=5`, and no label overlaps or clipping is present.

- [ ] **Step 4: Check repository integrity**

```bash
git diff --check
git status --short
git log -7 --oneline
```

Expected: no whitespace errors, no unexplained files, and commits for runner code, generated results, and documentation.
