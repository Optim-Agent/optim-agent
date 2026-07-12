# Anchor-Free Classification Cumulative Error Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible two-dataset cumulative-error verifier and run a
focused prompt-refinement loop until both GPT-5.5 cumulative-error ratios are
at most 0.8.

**Architecture:** A parent verifier orchestrates two existing example modules through isolated worker subprocesses and emits structured metrics. Autoresearch then changes only prompt-facing sampler behavior, using the worse dataset ratio as its primary metric and individual ratios as stop criteria.

**Tech Stack:** Python 3.12, PyTorch, Optuna, eight NVIDIA A100 GPUs, Codex CLI with GPT-5.5 medium.

## Global Constraints

- Use exactly 10 trials and seeds 0, 1, 2, 3, 4 for every candidate.
- Explicitly pass model `gpt-5.5` and effort `medium` for every agent run.
- Never inject benchmark-derived configurations, anchors, outcomes, or numeric priors.
- Tune only generic prompt construction and prompt-facing enrichment in `optim_agent`.
- Keep benchmark outputs and autoresearch control artifacts uncommitted.
- Run MNIST and CIFAR-10 concurrently on disjoint four-GPU sets, but evaluate prompt hypotheses serially.

---

### Task 1: Structured Cumulative-Error Harness

**Files:**
- Create: `scripts/verify_classification_cumulative_error.py`
- Modify: `tests/test_optim_agent.py`

**Interfaces:**
- Consumes: `examples.mnist.run`, `examples.cifar10.run`, and their curve JSON format.
- Produces: `_cumulative_error(root, dataset, label, seeds, trials) -> (float, list[float])`, `_metrics(root) -> dict`, and a final-line metrics JSON document.

- [ ] **Step 1: Write failing tests for cumulative-error aggregation and explicit model invocation**

```python
def test_verify_classification_cumulative_error():
    from scripts import verify_classification_cumulative_error as verify
    assert verify.TRIALS == 10
    assert verify.SEEDS == (0, 1, 2, 3, 4)
    assert verify.MODEL == "gpt-5.5"
    assert verify.EFFORT == "medium"
    assert verify._incumbent_error_curve([3.0, 4.0, 2.0]) == [3.0, 3.0, 2.0]
```

- [ ] **Step 2: Run the test and confirm the missing module causes the expected failure**

Run: `python tests/test_optim_agent.py`

Expected: failure while importing `scripts.verify_classification_cumulative_error`.

- [ ] **Step 3: Implement the minimal parent/worker verifier**

The worker redirects module globals before calling:

```python
module.run(method, list(SEEDS), TRIALS, EPOCHS, WORKERS, gpus,
           EFFORT, TIMEOUT, MODEL)
```

The parent starts MNIST and CIFAR-10 workers concurrently, validates five
complete 10-record curves per label, and prints `json.dumps(metrics)` as its
last line.

- [ ] **Step 4: Run focused and full offline verification**

Run: `python tests/test_optim_agent.py`

Expected: every listed check prints `ok` and the final line is
`all checks passed`.

- [ ] **Step 5: Commit the measurement harness**

```bash
git add scripts/verify_classification_cumulative_error.py tests/test_optim_agent.py
git commit -m "experiment: add anchor-free classification cumulative-error verifier"
```

### Task 2: Fresh Baseline And Run State

**Files:**
- Create uncommitted: `autoresearch-results/classification-n10-s5/**`
- Create uncommitted: `autoresearch-results/results.tsv`
- Create uncommitted: `autoresearch-results/state.json`
- Create uncommitted: `autoresearch-results/context.json`

**Interfaces:**
- Consumes: the verifier's final-line JSON.
- Produces: fixed Random/TPE curves and an initialized foreground autoresearch state whose primary metric is `max_ratio`.

- [ ] **Step 1: Run the full verifier on unchanged prompt code**

Run: `python scripts/verify_classification_cumulative_error.py`

Expected: final JSON includes finite `max_ratio`, `mnist_ratio`, and
`cifar10_ratio`, with five per-seed cumulative errors for each method.

- [ ] **Step 2: Initialize autoresearch after the baseline is known**

Run the bundled `autoresearch_init_run.py` with metric format `metrics_json`,
primary key `max_ratio`, direction `lower`, and acceptance criteria
`mnist_ratio <= 0.8` and `cifar10_ratio <= 0.8`.

- [ ] **Step 3: Run the guard**

Run: `python tests/test_optim_agent.py && python examples/mnist.py selfcheck && python examples/cifar10.py selfcheck`

Expected: all 18 or more offline checks pass and both self-checks print
`selfcheck ok`.

### Task 3: Prompt Refinement Loop

**Files:**
- Modify: `optim_agent/samplers.py`
- Modify: `tests/test_optim_agent.py`
- Read and preserve: `optim_agent/agent.py`

**Interfaces:**
- Consumes: retained prompt code, the last results rows, and generic current-run trial history.
- Produces: one trial commit and one mechanically logged decision per iteration.

- [ ] **Step 1: Select one generic prompt hypothesis using the four autoresearch perspectives**

Start with remaining-budget natural-language history, then quantitative
incumbent/delta presentation, then current-run TPE-style advisory evidence.
Reject any hypothesis containing benchmark-specific parameter values.

- [ ] **Step 2: Add one failing prompt-output test and observe the expected assertion failure**

Run: `python tests/test_optim_agent.py`

Expected: the new assertion fails because the selected prompt structure is
absent from `AgentSampler._prompt`.

- [ ] **Step 3: Implement only the selected prompt mechanism and make the offline test pass**

Run: `python tests/test_optim_agent.py`

Expected: `all checks passed`.

- [ ] **Step 4: Commit, run the full verifier, and run the guard after metric improvement**

Run: `python scripts/verify_classification_cumulative_error.py`

Expected: final-line JSON is parseable and reports both ratios. Run the guard
only when `max_ratio` improves.

- [ ] **Step 5: Keep or revert and record before choosing another hypothesis**

Keep only a justified improvement with a passing guard. Otherwise run
`git revert --no-edit HEAD`. Record the closed-out commit and metrics with the
bundled autoresearch helper before returning to Step 1. Stop only when both
individual ratios are at most 0.8 or a configured terminal condition occurs.
