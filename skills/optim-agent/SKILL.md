---
name: optim-agent
description: Use when the user wants to tune hyperparameters of the current project (learning rate, batch size, regularization, model settings, strategy parameters, etc.). You — the session agent — act as the sampler: read the code to understand what each hyperparameter does, then drive an optim-agent study, choosing each trial's configuration yourself.
---

# optim-agent — agent-in-the-loop hyperparameter tuning

In skill mode **you are the sampler**. Unlike the pip-package mode (where a
subprocess agent only sees the numeric history), you can read the project's
source, so exploit that: a learning rate interacts with batch size and
scheduler; a lookback window depends on data frequency; regularization depends
on model capacity. Reason qualitatively from the code AND quantitatively from
the trial history.

## Workflow

1. **Understand.** Read the training/entry script and locate every tunable
   hyperparameter. For each, note its role and a sensible range (log scale for
   rates and regularization strengths).
2. **Confirm with the user** which hyperparameters to tune, the objective
   metric, its direction, and the trial budget.
3. **Drive the study** with the ask/tell interface (`pip install optim-agent`
   if missing). Persist to JSON so the study resumes across sessions:

   ```python
   import optim_agent as oa

   study = oa.create_study(direction="minimize", storage="hpo_study.json")

   # each iteration: YOU choose params from code understanding + history below
   trial = study.ask({"lr": 3e-4, "batch_size": 64})
   value = run_training(**trial.params)   # however the project runs a trial
   study.tell(trial, value)
   print(study.best_value, study.best_params)
   ```

4. **Choose the next point** each iteration: review `study.trials` (params,
   value, state), balance exploration against exploitation, never repeat a
   point, and prefer moves you can justify from both the history and the code.
5. **Early-stop wasteful trials.** If the project reports intermediate metrics
   (per-epoch loss), record them with `trial.report(value, step)` and abandon
   clearly hopeless runs: `study.tell(trial, state="pruned")`.
6. **Report** the best configuration, the convergence trend, and one-line
   reasoning for why it wins.

## Rules

- Run trials one at a time; the whole point is that each choice uses all
  history so far.
- If a trial crashes, record it (`study.tell(trial, state="failed")`) and
  avoid that region.
- Keep the study JSON in the project root unless the user says otherwise.
