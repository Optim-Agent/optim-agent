"""Summary agent: an LLM-authored narration of the finished study.

`build_summary_prompt(study)` assembles the prompt (goal, context, search space,
trial history, best trial) following the `AgentSampler._prompt` pattern, reading
the study context via `getattr(study.sampler, "context", None)` so any sampler
works. `run_summary(study, backend, model, effort)` calls the agent once via
`agent.call_agent`, parses the reply with `agent.extract_json`, and validates
that every required section key is present — a reply that is unparseable *or*
missing sections falls back to the raw text, so a summary can never fail the
study. The summary is printed and returned (for the caller to persist); `None`
is returned (and nothing persisted) when there is no complete trial or the
agent call itself fails.

`backend="mock"` short-circuits before `call_agent` with a canned
structured-JSON reply built from the study, mirroring `AgentSampler._mock`, so
tests and demos run without a real agent CLI. `agent.BACKENDS`/`call_agent`
are unchanged.
"""

import json
import warnings

from . import agent as _agent

# Required section keys of the structured reply; a parsed reply missing any of
# them is treated as a parse failure and falls back to the raw text (R15).
SECTIONS = ("best_config", "space_insights", "trajectory_highlights", "next_steps")

_TITLES = {
    "best_config": "Best configuration",
    "space_insights": "Search space insights",
    "trajectory_highlights": "Trajectory highlights",
    "next_steps": "Suggested next steps",
}


def build_summary_prompt(study) -> str:
    """Prompt asking the agent to narrate the finished study as structured JSON."""
    best = study.best_trial
    context = getattr(study.sampler, "context", None)
    lines = [
        "You are an expert hyperparameter-optimization analyst. An optimization "
        "study has just finished; narrate the result for the user, thinking both "
        "qualitatively (what each parameter means) and quantitatively (the numbers).",
        "",
        f"Goal: {study.direction.upper()} the objective value.",
    ]
    if context:
        lines += ["", f"What is being tuned: {context}"]
    if study.space:
        lines += ["", "Search space:"]
        lines += [f"- {n}: {d.describe()}" for n, d in study.space.items()]
    complete = [t for t in study.trials if t.state == "complete" and t.value is not None]
    pruned = sum(1 for t in study.trials if t.state == "pruned")
    failed = sum(1 for t in study.trials if t.state == "failed")
    lines += ["", f"Trials: {len(study.trials)} total "
                  f"({len(complete)} complete, {pruned} pruned, {failed} failed)."]
    if best is not None:
        lines += [f"Best trial: #{best.number} value={best.value:.6g} params={best.params}"]
    if complete:
        ranked = sorted(complete, key=lambda t: t.value,
                        reverse=(study.direction == "maximize"))
        lines += ["", "Complete trials, best to worst (up to 10):"]
        for t in ranked[:10]:
            lines += [f"- #{t.number}: value={t.value:.6g}, params={t.params}"]
        lines += ["", "Most recent complete trials (up to 5):"]
        for t in complete[-5:]:
            lines += [f"- #{t.number}: value={t.value:.6g}, params={t.params}"]
    keys = ", ".join(f'"{k}": "..."' for k in SECTIONS)
    lines += ["",
              "Summarize the run: the best configuration and its value, what the "
              "results suggest about each search-space parameter, highlights of the "
              "optimization trajectory, and concrete suggested next steps.",
              f"Reply with ONLY a JSON object: {{{keys}}}."]
    return "\n".join(lines)


def _mock_reply(study) -> str:
    # ponytail: offline stand-in mirroring AgentSampler._mock — canned structured
    # JSON built from the study so tests/demos run without burning tokens.
    best = study.best_trial
    return json.dumps({
        "best_config": {"trial": best.number, "value": best.value, "params": best.params},
        "space_insights": "mock summary: no real agent was queried",
        "trajectory_highlights": f"{len(study.trials)} trials evaluated; "
                                 f"best is trial #{best.number}",
        "next_steps": "run more trials or narrow the space around the best point",
    })


def _render(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _format_summary(data) -> str:
    parts = ["[optim-agent] study summary"]
    for key in SECTIONS:
        parts += ["", f"{_TITLES[key]}:", _render(data[key])]
    return "\n".join(parts)


def run_summary(study, backend, model=None, effort=None):
    """Run the summary agent once and print the result.

    Returns the summary text for the caller to persist on the study. Returns
    `None` (persisting nothing) when there is no complete trial or the agent
    call fails. Replies that cannot be parsed into all required sections fall
    back to the raw reply text; this function never raises.
    """
    best = study.best_trial
    if best is None:
        print("[optim-agent] summary skipped: no complete trials")
        return None
    if backend == "mock":
        reply = _mock_reply(study)
    else:
        try:
            reply = _agent.call_agent(backend, model, build_summary_prompt(study),
                                      effort=effort)
        except Exception as e:
            warnings.warn(f"summary agent call failed ({e}); skipping summary")
            return None
    data = _agent.extract_json(reply)
    if data is None or not all(key in data for key in SECTIONS):
        summary = reply.strip()  # raw-text fallback: unparseable or partial reply
    else:
        summary = _format_summary(data)
    print(summary)
    return summary
