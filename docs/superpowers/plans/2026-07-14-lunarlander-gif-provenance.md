# LunarLander GIF Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate the public LunarLander GIF from the exact highest-scoring contextual GPT-5.5 case among HPO seeds 0--4.

**Architecture:** Add one selector that preserves the winning run's trial and training-seed provenance. Reuse the existing Q-learning evaluator, replay only its original evaluation seeds, and save the higher-return evaluation episode after confirming the replayed mean matches the committed artifact.

**Tech Stack:** Python standard library, Gymnasium, NumPy, imageio, pytest, Pillow for visual verification.

---

### Task 1: Specify the winning-case provenance

**Files:**
- Modify: `tests/test_optim_agent.py`
- Modify: `examples/rl_control.py`

- [ ] **Step 1: Add the failing selector test**

Add this test beside the existing RL-control tests:

```python
def test_rl_control_gif_selects_best_contextual_case():
    from examples import rl_control as rl

    run, trial, training_seed = rl._best_lunarlander_case()
    contextual_runs = [rl._load_artifact(rl.MODEL_LABEL, seed) for seed in rl.SEEDS]
    expected = max(
        contextual_runs,
        key=lambda item: item["best_values"]["LunarLander-v3"],
    )
    expected_trial = max(
        range(rl.N_TRIALS),
        key=expected["values"]["LunarLander-v3"].__getitem__,
    )

    assert run["method"] == rl.MODEL_LABEL
    assert run["seed"] == expected["seed"]
    assert trial == expected_trial
    assert training_seed == run["seed"] + trial
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python3 -m pytest tests/test_optim_agent.py::test_rl_control_gif_selects_best_contextual_case -q
```

Expected: FAIL with `AttributeError` because `_best_lunarlander_case` does not exist.

- [ ] **Step 3: Implement the minimal selector**

Add this helper immediately before `gif()`:

```python
def _best_lunarlander_case():
    runs = [_load_artifact(MODEL_LABEL, seed) for seed in SEEDS]
    run = max(
        runs,
        key=lambda item: item["best_values"]["LunarLander-v3"],
    )
    values = run["values"]["LunarLander-v3"]
    trial = max(range(N_TRIALS), key=values.__getitem__)
    return run, trial, run["seed"] + trial
```

- [ ] **Step 4: Run the test and verify GREEN**

Run:

```bash
python3 -m pytest tests/test_optim_agent.py::test_rl_control_gif_selects_best_contextual_case -q
```

Expected: `1 passed`.

### Task 2: Record the original highest-return evaluation episode

**Files:**
- Modify: `examples/rl_control.py`
- Modify: `docs/assets/lunarlander_policy.gif`

- [ ] **Step 1: Replace the seed-999 GIF path**

Update `gif()` to:

```python
run, trial, training_seed = _best_lunarlander_case()
params = run["params"][trial]
replayed_mean, q = _evaluate_env("LunarLander-v3", params, training_seed)
expected_mean = run["values"]["LunarLander-v3"][trial]
if not math.isclose(replayed_mean, expected_mean, rel_tol=0.0, abs_tol=1e-9):
    raise ValueError(
        f"LunarLander replay drift: expected {expected_mean}, got {replayed_mean}"
    )

env = gym.make("LunarLander-v3", render_mode="rgb_array")
bounds = _state_bounds("LunarLander-v3")
best_return = -math.inf
best_frames = None
best_eval_seed = None
for episode in range(EVAL_EPISODES):
    eval_seed = training_seed * 2000 + episode
    state, _ = env.reset(seed=eval_seed)
    frames = []
    total = 0.0
    done = False
    while not done and len(frames) < MAX_STEPS:
        frames.append(env.render())
        key = _discretize(state, bounds, int(params["bins"]))
        action = int(np.argmax(q.get(key, np.zeros(env.action_space.n))))
        state, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        done = terminated or truncated
    frames.append(env.render())
    if total > best_return:
        best_return = total
        best_frames = frames
        best_eval_seed = eval_seed
env.close()

output = ASSETS / "lunarlander_policy.gif"
imageio.mimsave(output, best_frames, fps=30)
print(
    f"wrote {output} (method={run['method']}, hpo_seed={run['seed']}, "
    f"trial={trial}, training_seed={training_seed}, "
    f"eval_seed={best_eval_seed}, return={best_return:.6f})"
)
```

Keep the existing import/exception boundary. Remove only the old cross-method
selection and fixed seed `999` replay.

- [ ] **Step 2: Run focused verification**

Run:

```bash
python3 -m pytest tests/test_optim_agent.py::test_rl_control_gif_selects_best_contextual_case -q
python3 examples/rl_control.py selfcheck
```

Expected: targeted test passes and self-check prints `RL control self-check passed`.

- [ ] **Step 3: Regenerate the GIF**

Run:

```bash
python3 examples/rl_control.py gif
```

Expected output includes `method=GPT-5.5`, `hpo_seed=0`, `trial=8`,
`training_seed=8`, `eval_seed=16001`, and return approximately `3.140071`.

### Task 3: Verify the artifact and repository

**Files:**
- Verify: `docs/assets/lunarlander_policy.gif`
- Verify: `examples/rl_control.py`
- Verify: `tests/test_optim_agent.py`

- [ ] **Step 1: Run the full guard**

Run:

```bash
python3 examples/credit_card.py selfcheck
python3 -m pytest tests/test_optim_agent.py -q
```

Expected: both commands pass with no new failures.

- [ ] **Step 2: Inspect GIF structure and key frames**

Use Pillow to verify that the GIF is animated, then render a temporary contact
sheet from its first, middle, and final frames.

Expected: the animation contains the exact seed-16001 episode, has no blank or
corrupt frames, and is visually consistent with the independently measured
return. It need not show a successful landing.

- [ ] **Step 3: Review the scoped diff**

Run:

```bash
git diff --check -- examples/rl_control.py tests/test_optim_agent.py
git status --short
```

Expected: only the selector/test/GIF changes from this task plus the user's
pre-existing `README.md` modification. Do not stage or alter `README.md`.
