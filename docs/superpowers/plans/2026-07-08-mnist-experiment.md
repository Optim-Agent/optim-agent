# MNIST Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible full-MNIST HPO example that uses 8 GPUs for parallel trials and writes JSON/PNG artifacts to `docs/assets`.

**Architecture:** One script, `examples/mnist.py`, follows the local example pattern with `download`, `run`, `plot`, and `selfcheck`. Helper functions are unit-tested without downloading MNIST; the full run uses PyTorch/torchvision and maps independent trials to GPUs.

**Tech Stack:** Python stdlib, optim-agent, PyTorch, torchvision, matplotlib for plotting.

## Global Constraints

- Use full torchvision MNIST train and test splits.
- Abort on dataset download failure; do not silently use partial or synthetic data.
- Save raw JSON and plotted figures under `docs/assets`.
- Prefer one script and small helpers over a multi-file experiment framework.
- Leave unrelated untracked files such as `paper/src/` untouched.

---

### Task 1: Test Pure MNIST Helpers

**Files:**
- Modify: `tests/test_optim_agent.py`

**Interfaces:**
- Consumes: planned functions in `examples.mnist`: `_best_error_curve(records)`, `_device_for_trial(number, gpus)`, `_sanitize_label(label)`, `_trial_record(trial, metrics)`.
- Produces: regression coverage for helper behavior that does not require MNIST data or CUDA.

- [ ] **Step 1: Write failing tests**

Add tests that import `examples.mnist` and assert:

```python
def test_mnist_helper_curves_and_labels():
    from examples import mnist

    assert mnist._sanitize_label("GPT-5.5") == "GPT-5.5"
    assert mnist._sanitize_label("agent/mock") == "agent-mock"
    assert mnist._best_error_curve([{"test_error": 9.0}, {"test_error": 7.5}, {"test_error": 8.0}]) == [9.0, 7.5, 7.5]
    assert mnist._device_for_trial(10, [0, 1, 2]) == "cuda:1"
    assert mnist._device_for_trial(3, []) == "cpu"
```

```python
def test_mnist_trial_record_serializes_metrics():
    from examples import mnist

    study = oa.create_study()
    trial = study.ask({"lr": 0.001, "batch_size": 128, "dropout": 0.2, "width": 32})
    trial.suggest_float("lr", 1e-4, 3e-2, log=True)
    trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    trial.suggest_float("dropout", 0.0, 0.6)
    trial.suggest_categorical("width", [16, 32, 64, 96, 128])
    metrics = {"test_error": 2.5, "test_acc": 97.5, "test_loss": 0.08, "history": [{"epoch": 1, "test_error": 2.5}]}

    rec = mnist._trial_record(trial, metrics)

    assert rec["params"]["batch_size"] == 128
    assert rec["test_error"] == 2.5
    assert rec["history"][0]["epoch"] == 1
```

- [ ] **Step 2: Run tests to verify red**

Run: `python tests/test_optim_agent.py`

Expected: import or attribute failure for missing `examples.mnist`.

### Task 2: Implement MNIST Example

**Files:**
- Create: `examples/mnist.py`

**Interfaces:**
- Produces: `_best_error_curve`, `_device_for_trial`, `_sanitize_label`, `_trial_record`, `download`, `run`, `plot`, `selfcheck`.

- [ ] **Step 1: Create minimal implementation**

Implement one script with:

- constants `ROOT`, `ASSETS`, `DATA`, `STORAGE`, `METHODS`;
- a compact `SmallCNN(width, dropout)`;
- full MNIST loaders using `torchvision.datasets.MNIST(..., download=<bool>)`;
- objective that tunes `lr`, `batch_size`, `dropout`, `width`;
- trial-to-device mapping by trial number modulo `--gpus`;
- JSON output file `mnist_curves_<method>_s<seed>.json`;
- plot output file `mnist_benchmarks.png`;
- `selfcheck` that validates helper functions and model forward shape.

- [ ] **Step 2: Run focused tests**

Run: `python tests/test_optim_agent.py`

Expected: all tests pass.

### Task 3: Verify Data, Plotting, and Full Run

**Files:**
- Generated: `data/mnist/`
- Generated: `docs/assets/mnist_curves_*.json`
- Generated: `docs/assets/mnist_benchmarks.png`

**Interfaces:**
- Consumes: `examples/mnist.py` CLI.
- Produces: verified dataset, results, and figure.

- [ ] **Step 1: Install missing examples dependencies if needed**

Run: `python -m pip install -e ".[examples]"`.

- [ ] **Step 2: Download full MNIST**

Run: `python examples/mnist.py download`

Expected: script reports MNIST train/test counts. If network fails, stop and ask the user to download manually.

- [ ] **Step 3: Run selfcheck**

Run: `python examples/mnist.py selfcheck`

Expected: `selfcheck ok`.

- [ ] **Step 4: Run the full 8-GPU experiment**

Run a bounded full-dataset benchmark, for example:

```bash
python examples/mnist.py run --method mock --seeds 0 1 2 --trials 24 --epochs 3 --workers 8 --gpus 0 1 2 3 4 5 6 7
python examples/mnist.py run --method Random --seeds 0 1 2 --trials 24 --epochs 3 --workers 8 --gpus 0 1 2 3 4 5 6 7
```

Expected: JSON files are written to `docs/assets`.

- [ ] **Step 5: Plot figures**

Run: `python examples/mnist.py plot`

Expected: `docs/assets/mnist_benchmarks.png` is written.

### Task 4: Final Verification

**Files:**
- Verify: all changed files.

- [ ] **Step 1: Run all local tests**

Run: `python tests/test_optim_agent.py`

Expected: all checks pass.

- [ ] **Step 2: Inspect git diff**

Run: `git diff -- examples/mnist.py tests/test_optim_agent.py docs/superpowers/specs/2026-07-08-mnist-experiment-design.md docs/superpowers/plans/2026-07-08-mnist-experiment.md`

Expected: diff only contains the MNIST example, tests, and planning docs.
