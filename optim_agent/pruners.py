"""AgentPruner: an LLM agent decides whether to stop an underperforming trial early."""

import warnings

from . import agent as _agent

# warmup: fraction of the longest completed curve before pruning is considered.
# interval: consult the agent every N reported steps (agent calls cost tokens).
LEVELS = {
    "loose":  dict(warmup=0.5,  interval=3,
                   stance="Only prune if the trial is almost certainly hopeless."),
    "medium": dict(warmup=0.3,  interval=2,
                   stance="Prune when the trial is clearly underperforming past trials."),
    "tight":  dict(warmup=0.15, interval=1,
                   stance="Prune aggressively at the first solid sign of underperformance."),
}


class AgentPruner:
    def __init__(self, backend="claude", model=None, level="medium", timeout=120):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {list(LEVELS)}")
        self.backend, self.model, self.level, self.timeout = backend, model, level, timeout

    def should_prune(self, study, trial) -> bool:
        cfg = LEVELS[self.level]
        done = [t for t in study.trials if t.state == "complete" and t.intermediate]
        if not done or not trial.intermediate:
            return False
        max_steps = max(len(t.intermediate) for t in done)
        step_idx = len(trial.intermediate)
        if step_idx < cfg["warmup"] * max_steps or step_idx % cfg["interval"] != 0:
            return False
        if self.backend == "mock":  # offline: prune if worse than the median final value
            finals = sorted(t.intermediate[-1][1] for t in done)
            median = finals[len(finals) // 2]
            cur = trial.intermediate[-1][1]
            return cur > median if study.direction == "minimize" else cur < median

        reference = sorted(done, key=lambda t: t.value,
                           reverse=(study.direction == "maximize"))[:5]
        lines = [
            "You are deciding whether to stop (prune) an in-progress hyperparameter "
            f"trial early. Goal: {study.direction.upper()} the objective.",
            "", "Intermediate curves of the best completed trials (step, value):",
        ]
        for t in reference:
            lines.append(f"- trial {t.number} (final {t.value:.6g}): {t.intermediate}")
        lines += ["", f"Current trial params: {trial.params}",
                  f"Current trial curve so far: {trial.intermediate}",
                  "", cfg["stance"],
                  'Reply with ONLY a JSON object: {"prune": true} or {"prune": false}.']
        try:
            reply = _agent.call_agent(self.backend, self.model, "\n".join(lines), self.timeout)
            data = _agent.extract_json(reply)
            return bool(data and data.get("prune") is True)
        except Exception as e:
            warnings.warn(f"pruner agent call failed ({e}); not pruning")
            return False  # never prune on error
