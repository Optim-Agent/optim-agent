"""Samplers. AgentSampler asks an LLM agent for the next point; effort trades tokens for depth."""

import random
import warnings

from . import agent as _agent

# Each effort level controls how rich the harness prompt is:
#   history    — how many past trials the agent sees (None = all)
#   reasoning  — ask the agent to reason about the landscape before answering
#   notes      — agent keeps a qualitative scratchpad carried across trials
#   candidates — agent proposes N ranked points, first valid one wins
EFFORTS = {
    "low":    dict(history=5,    reasoning=False, notes=False, candidates=1),
    "medium": dict(history=15,   reasoning=False, notes=False, candidates=1),
    "high":   dict(history=None, reasoning=True,  notes=False, candidates=1),
    "xhigh":  dict(history=None, reasoning=True,  notes=True,  candidates=1),
    "max":    dict(history=None, reasoning=True,  notes=True,  candidates=3),
}


class RandomSampler:
    """Uniform random baseline. Proposes nothing; suggest_* falls back to random draws."""

    def propose(self, study) -> dict:
        return {}


class AgentSampler:
    """LLM agent as the sampling strategy.

    backend: "claude" | "codex" | "opencode" | "mock" (offline, for tests/demos)
    model:   passed to the backend CLI's model flag; None = backend default
    effort:  one of EFFORTS, from "low" (terse, cheap) to "max" (full history,
             reasoning, persistent notes, multiple candidates)
    context: optional free-text description of what is being tuned, e.g.
             "learning rate and batch size of a CNN on MNIST"
    n_init:  random warmup trials before the agent is consulted
    """

    def __init__(self, backend="claude", model=None, effort="high", context=None,
                 n_init=2, timeout=300, seed=None):
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {list(EFFORTS)}")
        if backend != "mock" and backend not in _agent.BACKENDS:
            raise ValueError(f"backend must be one of {_agent.BACKENDS + ('mock',)}")
        self.backend, self.model, self.effort = backend, model, effort
        self.context, self.n_init, self.timeout = context, n_init, timeout
        self.rng = random.Random(seed)
        self.note = None  # qualitative scratchpad, fed back at xhigh/max efforts

    def propose(self, study) -> dict:
        done = [t for t in study.trials if t.state in ("complete", "pruned") and t.value is not None]
        if len([t for t in done if t.state == "complete"]) < self.n_init or not study.space:
            return {}
        if self.backend == "mock":
            return self._mock(study, done)
        cfg = EFFORTS[self.effort]
        prompt = self._prompt(study, done, cfg)
        for attempt in range(2):
            try:
                reply = _agent.call_agent(self.backend, self.model, prompt, self.timeout)
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
        lines += ["", "Search space:"]
        lines += [f"- {n}: {d.describe()}" for n, d in study.space.items()]

        # ponytail: cap "all history" at 400 rows so the prompt never blows past ARG_MAX;
        # pass prompts via stdin if someone really runs 10k-trial studies.
        shown = done[-(cfg["history"] or 400):]
        best = study.best_trial
        if best is not None and best not in shown:
            shown = [best] + shown
        lines += ["", "Trial history (oldest first):",
                  "| trial | " + " | ".join(names) + " | value | state |"]
        for t in shown:
            row = [str(t.params.get(n, "-")) for n in names]
            lines.append(f"| {t.number} | " + " | ".join(row) + f" | {t.value:.6g} | {t.state} |")
        if best is not None:
            lines += ["", f"Best so far: trial {best.number}, value={best.value:.6g}, params={best.params}"]
        if cfg["notes"] and self.note:
            lines += ["", f"Your notes from previous trials: {self.note}"]

        lines += ["", "Propose the next point to evaluate. Balance exploration of unvisited "
                      "regions against exploitation around promising ones; never repeat an "
                      "already-evaluated point exactly."]
        keys = ", ".join(f'"{n}": <value>' for n in names)
        if cfg["candidates"] > 1:
            lines += [f'Reply with ONLY a JSON object of the form '
                      f'{{"candidates": [{{{keys}}}, ...]}} containing your top '
                      f'{cfg["candidates"]} candidates ranked best first.']
        else:
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
        candidates = data.get("candidates") if isinstance(data.get("candidates"), list) else [data]
        for cand in candidates:
            if not isinstance(cand, dict) or not all(n in cand for n in study.space):
                continue
            try:
                return {n: dist.validate(cand[n]) for n, dist in study.space.items()}
            except (ValueError, TypeError):
                continue
        return None

    def _mock(self, study, done) -> dict:
        # ponytail: offline stand-in — gaussian jitter around the best point, ~hill climbing.
        # Exists so tests/demos run without burning tokens; not a real strategy.
        best = study.best_trial
        if best is None:
            return {}
        out = {}
        for name, dist in study.space.items():
            v = best.params.get(name)
            if v is None or self.rng.random() < 0.2:
                out[name] = dist.sample(self.rng)
            elif hasattr(dist, "low"):
                jitter = self.rng.gauss(0, 0.1 * (dist.high - dist.low))
                out[name] = dist.validate(v + jitter)
            else:
                out[name] = v
        return out
