# Credit Benchmark Single-Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the existing GPT-5.5 high-effort credit-default run as the sole effort-free agent configuration with history 20, explicit reasoning, and qualitative notes enabled.

**Architecture:** Keep the benchmark runner and artifact schema, but replace effort-derived method names with two fixed GPT-5.5 methods. Reuse the measured high-effort artifacts, delete obsolete effort variants, and make artifact validation pin the selected sampler settings.

**Tech Stack:** Python 3, pytest, JSON benchmark artifacts, matplotlib, Markdown.

---

### Task 1: Lock the fixed method and sampler contract

**Files:**
- Modify: `tests/test_optim_agent.py:320-365`
- Test: `tests/test_optim_agent.py`

- [ ] **Step 1: Change the protocol assertions to the fixed method set**

```python
assert credit.METHODS == (
    "Random", "TPE", "GPT-5.5", "GPT-5.5-no-context",
)
assert credit.SELECTED_METHOD == "GPT-5.5"
assert credit.AGENT_EFFORT == "high"
assert credit.HISTORY == 20
assert credit.EXPLICIT_REASONING is True
assert credit.QUALITATIVE_NOTES is True
```

- [ ] **Step 2: Assert the sampler receives the fixed settings**

```python
assert sampler.effort == "high"
assert sampler.history == 20
assert sampler.explicit_reasoning is True
assert sampler.qualitative_notes is True
```

- [ ] **Step 3: Run the focused tests and confirm failure**

Run: `python3 -m pytest tests/test_optim_agent.py -q -k 'credit_card_protocol or credit_default_split_model'`

Expected: failure because the implementation still exposes effort-suffixed methods and does not pass the fixed sampler options.

### Task 2: Simplify the benchmark implementation

**Files:**
- Modify: `examples/credit_card.py:50-290`
- Modify: `examples/credit_card.py:315-710`
- Test: `tests/test_optim_agent.py`

- [ ] **Step 1: Replace effort-derived constants with fixed settings**

```python
AGENT_EFFORT = "high"
HISTORY = 20
EXPLICIT_REASONING = True
QUALITATIVE_NOTES = True
NO_CONTEXT_METHOD = f"{MODEL_LABEL}-no-context"
METHODS = ("Random", "TPE", MODEL_LABEL, NO_CONTEXT_METHOD)
SELECTED_METHOD = MODEL_LABEL
```

- [ ] **Step 2: Return one fixed agent method spec**

```python
if method in (MODEL_LABEL, NO_CONTEXT_METHOD):
    return {
        "backend": BACKEND,
        "model": MODEL,
        "effort": AGENT_EFFORT,
        "use_context": method == MODEL_LABEL,
        "n_init": N_INIT,
    }
```

- [ ] **Step 3: Pass and persist the fixed sampler options**

```python
return oa.AgentSampler(
    ...,
    effort=AGENT_EFFORT,
    history=HISTORY,
    explicit_reasoning=EXPLICIT_REASONING,
    qualitative_notes=QUALITATIVE_NOTES,
)
```

Add `history`, `explicit_reasoning`, and `qualitative_notes` to agent artifact metadata so self-check rejects mismatched files.

- [ ] **Step 4: Remove unused effort styles and use only fixed method styles**

Keep plot entries for `Random`, `TPE`, `GPT-5.5`, and `GPT-5.5-no-context`.

- [ ] **Step 5: Run focused tests**

Run: `python3 -m pytest tests/test_optim_agent.py -q -k 'credit'`

Expected: implementation tests pass after artifact migration in Task 3; artifact-count tests may still fail before then.

### Task 3: Migrate committed artifacts and publication metadata

**Files:**
- Rename: `docs/assets/credit_default_GPT-5.5-high_s*.json` to `docs/assets/credit_default_GPT-5.5_s*.json`
- Rename: `docs/assets/credit_default_GPT-5.5-medium-no-context_s*.json` to `docs/assets/credit_default_GPT-5.5-no-context_s*.json`
- Delete: `docs/assets/credit_default_GPT-5.5-low_s*.json`
- Delete: `docs/assets/credit_default_GPT-5.5-medium_s*.json`
- Modify: `benchmarks/manifest.json:43-55`
- Modify: `tests/test_optim_agent.py:607-638`

- [ ] **Step 1: Rename retained files and delete obsolete files with `git mv` and `rm`**

Retain five contextual and five no-context artifacts, plus five Random and five TPE artifacts, for 20 total JSON files.

- [ ] **Step 2: Update retained artifact metadata mechanically**

For the contextual files set `method` to `GPT-5.5`; for controls set it to `GPT-5.5-no-context`. Set the protocol to the new fixed-config version and add:

```json
"history": 20,
"explicit_reasoning": true,
"qualitative_notes": true
```

Do not alter `values`, `params`, `best_params`, validation loss, or test loss.

- [ ] **Step 3: Pin the fixed settings in the manifest**

```json
"agent_effort": "high",
"history": 20,
"explicit_reasoning": true,
"qualitative_notes": true,
"selected_method": "GPT-5.5"
```

- [ ] **Step 4: Update publication tests to expect 20 artifacts and the fixed settings**

Run: `python3 -m pytest tests/test_optim_agent.py -q -k 'credit'`

Expected: all credit tests pass.

### Task 4: Update publication output and verify

**Files:**
- Modify: `README.md:241-269`
- Modify: `docs/assets/credit_card.png`

- [ ] **Step 1: Update the table from the retained high-effort artifact means**

Use validation `0.428395663` and test `0.422105924` for GPT-5.5. State that both are below TPE, while noting that selecting on test loss makes it a benchmark comparison rather than an untouched generalization estimate.

- [ ] **Step 2: Regenerate the plot**

Run: `python3 examples/credit_card.py plot`

Expected: `wrote .../docs/assets/credit_card.png`.

- [ ] **Step 3: Run artifact and test verification**

Run: `python3 examples/credit_card.py selfcheck`

Expected: `selfcheck ok: complete five-seed UCI credit-default artifacts are compatible`.

Run: `python3 -m pytest tests/test_optim_agent.py -q`

Expected: all available tests pass.

Run: `git diff --check`

Expected: no output.

- [ ] **Step 4: Review the final diff without staging unrelated README edits**

Confirm no MNIST/CIFAR artifacts changed and no autoresearch artifacts are staged.
