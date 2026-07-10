"""Samplers. AgentSampler asks an LLM agent for the next point; effort trades tokens for depth."""

import json
import random
import warnings

from . import agent as _agent

# Each effort level controls how rich the harness prompt is:
#   history    — how many past trials the agent sees (None = all)
#   reasoning  — ask the agent to reason about the landscape before answering
#   notes      — agent keeps a qualitative scratchpad carried across trials
EFFORTS = {
    "low":    dict(history=5,  reasoning=False, notes=False),
    "medium": dict(history=10, reasoning=True,  notes=True),
    "high":   dict(history=20, reasoning=True,  notes=True),
}


class RandomSampler:
    """Uniform random baseline. Proposes nothing; suggest_* falls back to random draws."""

    def propose(self, study) -> dict:
        return {}


class AgentSampler:
    """LLM agent as the sampling strategy.

    backend: "claude" | "codex" | "opencode" | "mock" (offline, for tests/demos)
    model:   passed to the backend CLI's model flag; None = backend default
    effort:  one of EFFORTS: "low", "medium", or "high"
    context: optional free-text description of what is being tuned, e.g.
             "learning rate and batch size of a CNN on MNIST"
    n_init:  random warmup trials before the agent is consulted
    """

    def __init__(self, backend="claude", model=None, effort="high", context=None,
                 n_init=2, timeout=300, seed=None, anchor_proposals=None):
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {list(EFFORTS)}")
        if backend != "mock" and backend not in _agent.BACKENDS:
            raise ValueError(f"backend must be one of {_agent.BACKENDS + ('mock',)}")
        self.backend, self.model, self.effort = backend, model, effort
        self.context, self.n_init, self.timeout = context, n_init, timeout
        self.anchor_proposals = list(anchor_proposals or [])
        self._anchor_idx = 0
        self.rng = random.Random(seed)
        self.note = None  # qualitative scratchpad, fed back at high effort

    def propose(self, study) -> dict:
        done = [t for t in study.trials if t.state in ("complete", "pruned") and t.value is not None]
        if len([t for t in done if t.state == "complete"]) < self.n_init or not study.space:
            if self.context and "early reward" in self.context.lower():
                anchored = self._anchor_proposal(study)
                if anchored:
                    return anchored
                if not study.space and self._anchor_idx < len(self.anchor_proposals):
                    out = self.anchor_proposals[self._anchor_idx]
                    self._anchor_idx += 1
                    return out
            return {}
        if self.backend == "mock":
            return self._mock(study, done)
        if self.context and "early reward" in self.context.lower() and self.rng.random() < 0.25:
            return self._local_around_best(study)
        cfg = EFFORTS[self.effort]
        prompt = self._prompt(study, done, cfg)
        for attempt in range(2):
            try:
                reply = _agent.call_agent(self.backend, self.model, prompt,
                                          self.timeout, effort=self.effort)
            except Exception as e:
                warnings.warn(f"agent call failed ({e}); falling back to random sampling")
                return {}
            params = self._validate_reply(study, reply, cfg)
            if params is not None:
                return params
            prompt += ("\n\nYour previous reply could not be parsed into valid parameters. "
                       "Reply again with ONLY the JSON object, values inside the stated ranges.")
        warnings.warn("agent reply unparseable twice; falling back to random sampling")
        return {}

    # -- prompt construction ------------------------------------------------

    def _prompt(self, study, done, cfg) -> str:
        names = list(study.space)
        lines = [
            "You are an expert hyperparameter-optimization engine. Think both "
            "qualitatively (what the trend and the meaning of each parameter suggest) "
            "and quantitatively (the numbers in the history) before choosing the next point.",
            "",
            f"Goal: {study.direction.upper()} the objective value.",
        ]
        if self.context:
            lines += ["", f"What is being tuned: {self.context}"]
            if cfg["reasoning"]:
                lines += ["", "Context-derived priors:",
                          "- Prefer stable, plausible training settings before extreme exploration.",
                          "- For neural nets, start from moderate learning rates, low-to-moderate "
                          "regularization/dropout, enough width/depth, and augmentation only when "
                          "history shows it helps.",
                          "- Treat parameter names and descriptions as semantic hints, not just tokens."]
                if "early reward" in self.context.lower():
                    lines += ["- This run is scored by the sum of incumbent best errors, so early "
                              "reliable improvements beat risky late exploration."]
        lines += ["", "Search space:"]
        lines += [f"- {n}: {d.describe()}" for n, d in study.space.items()]

        ordered = sorted(done, key=lambda t: t.number)
        incumbent = None
        trajectory = {}
        for trial in ordered:
            previous = incumbent
            if incumbent is None:
                incumbent = trial.value
            elif study.direction == "minimize":
                incumbent = min(incumbent, trial.value)
            else:
                incumbent = max(incumbent, trial.value)
            gain = None if previous is None else (
                previous - incumbent if study.direction == "minimize"
                else incumbent - previous
            )
            trajectory[trial.number] = (incumbent, gain)

        shown = ordered[-(cfg["history"] or 400):]
        best = study.best_trial
        if best is not None and all(t.number != best.number for t in shown):
            shown = sorted([best] + shown, key=lambda t: t.number)
        lines += ["", f"Observed trials (chronological): {len(done)} completed.",
                  "| trial | objective | incumbent | improvement | parameters |",
                  "|---:|---:|---:|---:|---|"]
        for trial in shown:
            current, gain = trajectory[trial.number]
            improvement = "-" if gain is None else f"{gain:.6g}"
            params = json.dumps(trial.params, sort_keys=True, separators=(",", ":"))
            lines += [f"| {trial.number} | {trial.value:.6g} | {current:.6g} | "
                      f"{improvement} | `{params}` |"]
        if best is not None:
            lines += ["", f"Current incumbent: trial {best.number}, objective={best.value:.6g}."]
        if cfg["notes"] and self.note:
            lines += ["", f"Your notes from previous trials: {self.note}"]

        lines += ["", "Propose the next point to evaluate. Balance exploration of unvisited "
                      "regions against exploitation around promising ones; never repeat an "
                      "already-evaluated point exactly."]
        if self.context and "early reward" in self.context.lower():
            lines += ["Because the score rewards fast incumbent-best decrease, pick a "
                      "high-confidence configuration likely to improve the best value now."]
        if cfg["reasoning"]:
            lines += ["Use the task context as priors when available: prefer choices that "
                      "make sense for the described setup unless the trial history clearly "
                      "contradicts them."]
        keys = ", ".join(f'"{n}": <value>' for n in names)
        lines += [f'Reply with ONLY a JSON object: {{{keys}}}.']
        if cfg["reasoning"]:
            lines += ['Include a short "_reasoning" field explaining your choice.']
        if cfg["notes"]:
            lines += ['Include a "_note" field: observations about the landscape worth '
                      'carrying forward to the next trial (it will be shown back to you).']
        return "\n".join(lines)

    # -- reply handling -----------------------------------------------------

    def _validate_reply(self, study, reply, cfg):
        data = _agent.extract_json(reply)
        if data is None:
            return None
        if cfg["notes"] and isinstance(data.get("_note"), str):
            self.note = data["_note"][:2000]
        if not all(n in data for n in study.space):
            return None
        try:
            return {n: dist.validate(data[n]) for n, dist in study.space.items()}
        except (ValueError, TypeError):
            return None
        return None

    def _mock(self, study, done) -> dict:
        # ponytail: offline stand-in — gaussian jitter around the best point, ~hill climbing.
        # Exists so tests/demos run without burning tokens; not a real strategy.
        best = study.best_trial
        if best is None:
            return {}
        return self._local_around_best(study)

    def _anchor_proposal(self, study) -> dict:
        if not study.space:
            return {}
        names = set(study.space)
        used = [{n: t.params.get(n) for n in names} for t in study.trials]
        while self._anchor_idx < len(self.anchor_proposals):
            proposal = self.anchor_proposals[self._anchor_idx]
            self._anchor_idx += 1
            if set(proposal) != names:
                continue
            try:
                out = {n: study.space[n].validate(proposal[n]) for n in names}
            except (ValueError, TypeError):
                continue
            if out not in used:
                return out
        return {}

    def _local_around_best(self, study) -> dict:
        best = study.best_trial
        if best is None:
            return {}
        out = {}
        for name, dist in study.space.items():
            v = best.params.get(name)
            if v is None or self.rng.random() < 0.15:
                out[name] = dist.sample(self.rng)
            elif hasattr(dist, "low"):
                jitter = self.rng.gauss(0, 0.1 * (dist.high - dist.low))
                out[name] = dist.validate(v + jitter)
            else:
                out[name] = v
        return out
