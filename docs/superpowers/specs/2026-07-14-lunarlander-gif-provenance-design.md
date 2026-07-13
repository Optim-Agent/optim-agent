# LunarLander GIF Provenance Design

## Goal

Regenerate `docs/assets/lunarlander_policy.gif` from the highest-scoring
contextual GPT-5.5 LunarLander case among HPO seeds 0--4, using the exact
training and evaluation seeds that produced the committed benchmark score.

## Current Failure

`gif()` currently selects the best committed agent artifact, but then discards
its trial provenance. It retrains the selected parameters with seed `999` and
records a rollout reset with seed `999`. The resulting GIF therefore does not
reproduce the selected artifact and currently shows a crash with return
`-217.51`.

The function also searches both contextual and no-context artifacts, which is
ten runs rather than the requested five contextual seeds.

## Selection Contract

1. Load only `GPT-5.5` contextual artifacts for seeds 0--4.
2. Select the run with the largest committed
   `best_values["LunarLander-v3"]`.
3. Within that run, recover the trial index whose recorded LunarLander value is
   maximal.
4. Recover the actual environment-training seed as
   `run["seed"] + trial_index`, matching `_objective_for()`.
5. Retrain the Q-table with that seed and verify that its replayed mean equals
   the committed best value within `1e-9` absolute tolerance.
6. Replay the policy on the same `EVAL_EPISODES` reset seeds used by
   `_evaluate_env()`: `training_seed * 2000 + episode`.
7. Save the frames from the evaluation episode with the largest return.

For the current artifacts this selects HPO seed 0, trial 8, training seed 8,
and evaluation reset seed 16001. Its single-episode return is approximately
`3.14`. This is the highest episode for the best five-seed case, but it is not a
successful landing; the GIF must represent the data faithfully rather than
imply success.

## Code Shape

Add one small selector in `examples/rl_control.py` that returns the selected
run, trial index, and training seed. `gif()` uses that selector, performs the
deterministic replay check, evaluates both original evaluation episodes while
capturing frames, and writes only the higher-return animation. The completion
message includes method, HPO seed, trial, training seed, evaluation seed, and
return.

No benchmark values, JSON artifacts, search behavior, README claims, or other
methods change.

## Failure Handling

- Missing or invalid artifacts continue through the existing clear skip path.
- A replayed mean that differs from the committed value aborts before the GIF
  is overwritten.
- GIF output is written only after a complete evaluation episode is selected.

## Verification

1. Add a regression test that proves selection is restricted to the five
   contextual artifacts and that the training seed includes the winning trial
   index.
2. Run the test before implementation and confirm it fails because the selector
   is absent.
3. Implement the selector and GIF replay change, then run the targeted test and
   the existing RL self-check.
4. Run the full repository guard.
5. Regenerate the GIF, inspect key frames, and independently replay its reported
   evaluation seed to confirm the return and provenance.
