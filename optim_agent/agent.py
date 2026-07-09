"""Thin layer over agent CLIs (claude / codex / opencode): build command, run, parse JSON reply."""

import json
import re
import subprocess

BACKENDS = ("claude", "codex", "opencode")


# each CLI's own reasoning-effort flag.
def _effort_flag(backend: str, effort: str) -> list:
    if backend == "codex":
        return ["-c", f"model_reasoning_effort={effort}"]
    return {"claude": ["--effort", effort], "opencode": ["--variant", effort]}[backend]


def _cmd(backend: str, model, prompt: str, effort=None) -> list:
    if backend == "claude":
        cmd = ["claude", "-p"]
        if model:
            cmd += ["--model", model]
    elif backend == "codex":
        cmd = ["codex", "exec", "--skip-git-repo-check"]
        if model:
            cmd += ["-m", model]
    elif backend == "opencode":
        cmd = ["opencode", "run"]
        if model:
            cmd += ["-m", model]
    else:
        raise ValueError(f"unknown backend {backend!r}, expected one of {BACKENDS}")
    if effort:
        cmd += _effort_flag(backend, effort)
    return cmd + [prompt]


def call_agent(backend: str, model, prompt: str, timeout: float = 300, effort=None) -> str:
    proc = subprocess.run(
        _cmd(backend, model, prompt, effort),
        capture_output=True, text=True, timeout=timeout,
        stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{backend} exited {proc.returncode}: {proc.stderr.strip()[-500:]}")
    return proc.stdout


def extract_json(text: str):
    """Return the last JSON object found in agent output, or None.

    Agents wrap answers in prose/fences and sometimes echo the schema first,
    so we collect every decodable object and keep the last one.
    """
    text = text[-100_000:]  # the answer is at the end; don't O(n^2)-scan huge CLI logs
    candidates = []
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S):
        candidates.append(m.group(1))
    decoder = json.JSONDecoder()
    i = 0
    while True:
        i = text.find("{", i)
        if i == -1:
            break
        try:
            obj, end = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                candidates.append(text[i:i + end])
                i += end
                continue
        except ValueError:
            pass
        i += 1
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            continue
    return None
