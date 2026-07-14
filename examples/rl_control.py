"""CPU-only Gymnasium RL hyperparameter benchmark.

    python examples/rl_control.py preflight
    python examples/rl_control.py run
    python examples/rl_control.py selfcheck
    python examples/rl_control.py summary
    python examples/rl_control.py plot
    python examples/rl_control.py gif
"""

import argparse
import json
import math
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optim_agent as oa
from optim_agent import agent as agent_api


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
BACKEND = "codex"
MODEL = "gpt-5.5"
MODEL_LABEL = "GPT-5.5"
AGENT_EFFORT = "high"
AGENT_HISTORY = 5
SEEDS = (0, 1, 2, 3, 4)
N_TRIALS = 20
N_INIT = 3
ENVS = ("Acrobot-v1", "LunarLander-v3")
TRAIN_EPISODES = (8, 16, 24, 32)
EVAL_EPISODES = 2
MAX_STEPS = 200
SCHEMA_VERSION = 1
PROTOCOL_VERSION = "rl-control-qlearning-v2"
OBJECTIVE = "mean evaluation return"
METHODS = ("Random", "TPE", MODEL_LABEL, f"{MODEL_LABEL}-no-context")
TASK_CONTEXT = (
    "Tune one CPU tabular/discretized Q-learning configuration shared by Gymnasium "
    "Acrobot-v1 and LunarLander-v3. The scalar objective averages their evaluation "
    "returns, so improve both rather than sacrificing one environment. This is tabular "
    "Q-learning backed by a dictionary, not a neural network: only visited discrete states "
    "allocate rows. Acrobot gives roughly -1 per step until the terminal height is reached "
    "and is capped at 200 steps; escaping sooner is better. LunarLander has shaped returns "
    "for controlled descent, stable contact, fuel use, and crashes. Treat the numeric trial "
    "history as authoritative; use the descriptions only to explain mechanisms, not to rule "
    "out high-resolution or aggressive settings that perform well."
)
PARAMETER_CONTEXT = {
    "learning_rate": (
        "Q-learning update step size. Larger values adapt quickly from few visits but can "
        "overwrite estimates; smaller values average repeated visits. Let observed returns "
        "decide which regime works for each seed."
    ),
    "gamma": (
        "Discount factor. Acrobot needs credit across a long swing-up sequence and LunarLander "
        "must plan a controlled descent. High values retain long-horizon information, while "
        "lower values emphasize immediate shaped feedback."
    ),
    "epsilon_decay": (
        "Exploration multiplier applied after every training episode. Values near 1 preserve "
        "exploration across the tiny episode budget; lower values switch quickly to the current "
        "greedy policy."
    ),
    "min_epsilon": (
        "Minimum exploration probability during training. A larger floor keeps discovering new "
        "maneuvers; a smaller floor consolidates the current policy. Evaluation is greedy."
    ),
    "bins": (
        "Uniform bins per state dimension. A theoretical dense table grows exponentially, but "
        "this dictionary stores only visited states. Fewer bins share experience broadly; more "
        "bins preserve control precision, so do not reject 10-12 bins on memory grounds."
    ),
    "train_episodes": (
        "Training episodes per environment and trial. More episodes increase state-action coverage "
        "and cost CPU; fewer episodes can work when updates and exploration are aggressive."
    ),
}
LANDING_HPO_SEED = 0
LANDING_EVAL_EPISODES = 20
LANDING_MAX_TRIALS = 100
LANDING_AGENT_HISTORY = 20
LANDING_ARTIFACT = ASSETS / "lunarlander_landing.json"
LANDING_REFERENCE_PARAMS = {
    "x_position_gain": 0.5,
    "x_velocity_gain": 1.0,
    "angle_gain": 0.5,
    "angular_velocity_gain": 1.0,
    "hover_x_gain": 0.55,
    "hover_gain": 0.5,
    "vertical_velocity_gain": 0.5,
}
LANDING_PARAMETER_CONTEXT = {
    "x_position_gain": "Horizontal-position contribution to the target tilt toward the pad.",
    "x_velocity_gain": "Horizontal-velocity damping contribution to the target tilt.",
    "angle_gain": "Correction strength from target tilt to current angle.",
    "angular_velocity_gain": "Angular-rate damping used to prevent oscillation near touchdown.",
    "hover_x_gain": "Desired altitude contribution from horizontal distance to the pad.",
    "hover_gain": "Vertical-position correction strength.",
    "vertical_velocity_gain": "Vertical-speed damping used to reduce touchdown velocity.",
}
LANDING_TASK_CONTEXT = (
    "Tune a deterministic discrete LunarLander-v3 heuristic controller for a public GIF. "
    "One HPO seed evaluates every configuration on the same 20 environment seeds. A rollout "
    "is a successful landing only when Gymnasium terminates because the lander body sleeps, "
    "which produces a final reward of +100. Prefer configurations that land, then higher mean "
    "return. The controller targets the pad using position and velocity feedback; excessive "
    "gains oscillate or crash, while weak gains fail to arrest descent."
)


def _method_spec(method):
    if method == "Random":
        return {"backend": None, "model": None, "use_context": False, "n_init": 0}
    if method == "TPE":
        return {"backend": "tpe", "model": None, "use_context": False, "n_init": N_INIT}
    if method == MODEL_LABEL:
        return {"backend": BACKEND, "model": MODEL, "use_context": True, "n_init": N_INIT}
    if method == f"{MODEL_LABEL}-no-context":
        return {"backend": BACKEND, "model": MODEL, "use_context": False, "n_init": N_INIT}
    raise ValueError(f"unknown RL control method: {method}")


def _artifact_path(method, seed):
    return ASSETS / f"rl_control_{method}_s{seed}.json"


def _common_metadata(method, seed):
    spec = _method_spec(method)
    is_agent = spec["backend"] == BACKEND
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_VERSION,
        "method": method,
        "backend": spec["backend"],
        "model": spec["model"],
        "agent_effort": AGENT_EFFORT if is_agent else None,
        "history": AGENT_HISTORY if is_agent else None,
        "parameter_context": PARAMETER_CONTEXT if is_agent and spec["use_context"] else None,
        "explicit_reasoning": method != MODEL_LABEL if is_agent else None,
        "qualitative_notes": method != MODEL_LABEL if is_agent else None,
        "use_context": spec["use_context"] if is_agent else None,
        "context_policy": (
            "supplied task and parameter context"
            if is_agent and spec["use_context"]
            else "no supplied task context" if is_agent else "not applicable"
        ),
        "task_context": TASK_CONTEXT if is_agent and spec["use_context"] else None,
        "agent_failure_policy": "fail_closed" if is_agent else "not applicable",
        "n_init": spec["n_init"],
        "seed": seed,
        "trials": N_TRIALS,
        "environments": list(ENVS),
        "objective": OBJECTIVE,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _valid_number(value, low=None, high=None):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and (low is None or value >= low)
        and (high is None or value <= high)
    )


def _valid_params(params):
    expected = set(PARAMETER_CONTEXT)
    if not isinstance(params, list) or len(params) != N_TRIALS:
        return False
    for point in params:
        if not isinstance(point, dict) or set(point) != expected:
            return False
        if not _valid_number(point["learning_rate"], 0.02, 0.8):
            return False
        if not _valid_number(point["gamma"], 0.85, 0.999):
            return False
        if not _valid_number(point["epsilon_decay"], 0.90, 0.999):
            return False
        if not _valid_number(point["min_epsilon"], 0.01, 0.30):
            return False
        if point["bins"] not in (6, 8, 10, 12):
            return False
        if point["train_episodes"] not in TRAIN_EPISODES:
            return False
    return True


def _has_valid_creation_time(run):
    value = run.get("created_at")
    if not isinstance(value, str):
        return False
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return False
    return timestamp.tzinfo is not None and timestamp.utcoffset() is not None


def _validate_artifact(run, method, seed):
    expected = _common_metadata(method, seed)
    expected.pop("created_at")
    valid = all(run.get(key) == value for key, value in expected.items())
    valid = valid and _has_valid_creation_time(run)
    values = run.get("values")
    valid = valid and isinstance(values, dict) and set(values) == set(ENVS)
    valid = valid and all(
        isinstance(series, list)
        and len(series) == N_TRIALS
        and all(_valid_number(value) for value in series)
        for series in (values or {}).values()
    )
    valid = valid and _valid_params(run.get("params"))
    if valid:
        best_values = run.get("best_values")
        best_params = run.get("best_params")
        valid = isinstance(best_values, dict) and set(best_values) == set(ENVS)
        valid = valid and isinstance(best_params, dict) and set(best_params) == set(ENVS)
        for env_name in ENVS:
            best_index = max(range(N_TRIALS), key=values[env_name].__getitem__)
            valid = valid and math.isclose(
                best_values[env_name],
                values[env_name][best_index],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            valid = valid and best_params[env_name] == run["params"][best_index]
    valid = valid and _valid_number(run.get("elapsed_seconds"), 0.0)
    if not valid:
        raise ValueError(f"incompatible RL control artifact for {method} seed {seed}")
    return run


def _incumbent(values):
    best = -math.inf
    curve = []
    for value in values:
        best = max(best, value)
        curve.append(best)
    return curve


def _gymnasium():
    try:
        import gymnasium as gym
        import numpy as np
    except ImportError as error:
        raise SystemExit('Install the RL extra: pip install -e ".[rl,examples]"') from error
    return gym, np


def _context(name, use_context):
    return PARAMETER_CONTEXT[name] if use_context else None


def suggest_params(trial, use_context=True):
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.02, 0.8, log=True,
            context=_context("learning_rate", use_context),
        ),
        "gamma": trial.suggest_float(
            "gamma", 0.85, 0.999,
            context=_context("gamma", use_context),
        ),
        "epsilon_decay": trial.suggest_float(
            "epsilon_decay", 0.90, 0.999,
            context=_context("epsilon_decay", use_context),
        ),
        "min_epsilon": trial.suggest_float(
            "min_epsilon", 0.01, 0.30, log=True,
            context=_context("min_epsilon", use_context),
        ),
        "bins": trial.suggest_categorical(
            "bins", (6, 8, 10, 12),
            context=_context("bins", use_context),
        ),
        "train_episodes": trial.suggest_categorical(
            "train_episodes", TRAIN_EPISODES,
            context=_context("train_episodes", use_context),
        ),
    }


def suggest_landing_params(trial):
    ranges = {
        "x_position_gain": (0.2, 0.9),
        "x_velocity_gain": (0.5, 1.5),
        "angle_gain": (0.2, 0.9),
        "angular_velocity_gain": (0.5, 1.6),
        "hover_x_gain": (0.25, 0.9),
        "hover_gain": (0.25, 0.9),
        "vertical_velocity_gain": (0.25, 0.9),
    }
    return {
        name: trial.suggest_float(
            name, low, high, context=LANDING_PARAMETER_CONTEXT[name],
        )
        for name, (low, high) in ranges.items()
    }


def _landing_action(state, params):
    x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = state
    target_angle = x * params["x_position_gain"] + vx * params["x_velocity_gain"]
    target_angle = min(0.4, max(-0.4, target_angle))
    target_hover = params["hover_x_gain"] * abs(x)
    angle_control = (
        (target_angle - angle) * params["angle_gain"]
        - angular_velocity * params["angular_velocity_gain"]
    )
    hover_control = (
        (target_hover - y) * params["hover_gain"]
        - vy * params["vertical_velocity_gain"]
    )
    if left_leg or right_leg:
        angle_control = 0.0
        hover_control = -vy * params["vertical_velocity_gain"]
    if hover_control > abs(angle_control) and hover_control > 0.05:
        return 2
    if angle_control < -0.05:
        return 3
    if angle_control > 0.05:
        return 1
    return 0


def _is_successful_landing(terminated, truncated, final_reward):
    return bool(terminated and not truncated and final_reward == 100.0)


def _evaluate_landing_params(params, seeds):
    gym, _ = _gymnasium()
    env = gym.make("LunarLander-v3")
    episodes = []
    for seed in seeds:
        state, _ = env.reset(seed=seed)
        total = 0.0
        final_reward = None
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            state, reward, terminated, truncated, _ = env.step(
                _landing_action(state, params)
            )
            total += float(reward)
            final_reward = float(reward)
            steps += 1
        episodes.append({
            "seed": seed,
            "return": total,
            "landed": _is_successful_landing(terminated, truncated, final_reward),
            "final_reward": final_reward,
            "steps": steps,
        })
    env.close()
    return episodes


def _validate_landing_budget(max_trials):
    if not 1 <= max_trials <= LANDING_MAX_TRIALS:
        raise ValueError("LunarLander animation HPO allows at most 100 trials")


def _state_bounds(env_name):
    if env_name == "Acrobot-v1":
        return [
            (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
            (-1.0, 1.0), (-12.57, 12.57), (-28.27, 28.27),
        ]
    if env_name == "LunarLander-v3":
        return [
            (-1.5, 1.5), (-1.5, 1.5), (-5.0, 5.0), (-5.0, 5.0),
            (-3.2, 3.2), (-5.0, 5.0), (0.0, 1.0), (0.0, 1.0),
        ]
    raise ValueError(f"unknown environment: {env_name}")


def _discretize(state, bounds, bins):
    result = []
    for value, (low, high) in zip(state, bounds):
        clipped = min(max(float(value), low), high)
        bucket = int((clipped - low) / (high - low) * bins)
        result.append(min(bins - 1, max(0, bucket)))
    return tuple(result)


def _evaluate_env(env_name, params, seed):
    gym, np = _gymnasium()
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    bounds = _state_bounds(env_name)
    actions = env.action_space.n
    rng = np.random.default_rng(seed)
    q = {}
    epsilon = 1.0

    for episode in range(int(params["train_episodes"])):
        env.action_space.seed(seed * 3000 + episode)
        state, _ = env.reset(seed=seed * 1000 + episode)
        key = _discretize(state, bounds, int(params["bins"]))
        done = False
        steps = 0
        # ponytail: capped episodes keep the CPU demo runnable; raise caps for stronger RL claims.
        while not done and steps < MAX_STEPS:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q.get(key, np.zeros(actions))))
            next_state, signal, terminated, truncated, _ = env.step(action)
            next_key = _discretize(next_state, bounds, int(params["bins"]))
            row = q.setdefault(key, np.zeros(actions))
            done = terminated or truncated
            future = 0.0 if done else np.max(q.get(next_key, np.zeros(actions)))
            target = signal + params["gamma"] * future
            row[action] += params["learning_rate"] * (target - row[action])
            key = next_key
            steps += 1
        epsilon = max(params["min_epsilon"], epsilon * params["epsilon_decay"])

    returns = []
    for episode in range(EVAL_EPISODES):
        state, _ = eval_env.reset(seed=seed * 2000 + episode)
        key = _discretize(state, bounds, int(params["bins"]))
        done = False
        total = 0.0
        steps = 0
        while not done and steps < MAX_STEPS:
            action = int(np.argmax(q.get(key, np.zeros(actions))))
            state, signal, terminated, truncated, _ = eval_env.step(action)
            key = _discretize(state, bounds, int(params["bins"]))
            total += float(signal)
            done = terminated or truncated
            steps += 1
        returns.append(total)
    env.close()
    eval_env.close()
    return float(np.mean(returns)), q


def _evaluate_params(params, seed):
    return {env_name: _evaluate_env(env_name, params, seed)[0] for env_name in ENVS}


class OptunaTrialAdapter:
    def __init__(self, trial):
        self.trial = trial

    @property
    def number(self):
        return self.trial.number

    def suggest_float(self, name, low, high, *, log=False, context=None):
        return self.trial.suggest_float(name, low, high, log=log)

    def suggest_categorical(self, name, choices, *, context=None):
        return self.trial.suggest_categorical(name, choices)


def _make_sampler(method, seed, timeout, agent_cwd):
    spec = _method_spec(method)
    if method == "Random":
        return oa.RandomSampler()
    if method == "TPE":
        raise ValueError("TPE uses Optuna directly")
    return oa.AgentSampler(
        backend=spec["backend"],
        model=spec["model"],
        effort=AGENT_EFFORT,
        context=TASK_CONTEXT if spec["use_context"] else None,
        n_init=spec["n_init"],
        timeout=timeout,
        seed=seed,
        agent_cwd=agent_cwd,
        fail_closed=True,
        history=AGENT_HISTORY,
        explicit_reasoning=method != MODEL_LABEL,
        qualitative_notes=method != MODEL_LABEL,
    )


def _objective_for(method, seed, records):
    use_context = _method_spec(method)["use_context"]

    def objective(trial):
        params = suggest_params(trial, use_context=use_context)
        scores = _evaluate_params(params, seed + trial.number)
        records.append((trial.number, params, scores))
        return sum(scores.values()) / len(scores)

    return objective


def _run_search(method, seed, timeout, workers):
    records = []
    if method == "TPE":
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=N_INIT),
        )
        study.optimize(
            lambda trial: _objective_for(method, seed, records)(OptunaTrialAdapter(trial)),
            n_trials=N_TRIALS,
            n_jobs=workers,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="optim-agent-rl-") as agent_cwd:
            sampler = _make_sampler(method, seed, timeout, agent_cwd)
            study = oa.create_study(
                direction="maximize", sampler=sampler, seed=seed, max_concurrency=workers,
            )
            study.optimize(_objective_for(method, seed, records), n_trials=N_TRIALS)

    records.sort(key=lambda item: item[0])
    values = {env_name: [scores[env_name] for _, _, scores in records] for env_name in ENVS}
    params = [point for _, point, _ in records]
    return values, params


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1) + "\n")
    temporary.replace(path)


def run_landing_hpo(max_trials=LANDING_MAX_TRIALS, timeout=600):
    _validate_landing_budget(max_trials)
    eval_seeds = tuple(range(LANDING_EVAL_EPISODES))
    records = []

    with tempfile.TemporaryDirectory(prefix="optim-agent-lunarlander-") as agent_cwd:
        sampler = oa.AgentSampler(
            backend=BACKEND,
            model=MODEL,
            effort=AGENT_EFFORT,
            context=LANDING_TASK_CONTEXT,
            n_init=0,
            timeout=timeout,
            seed=LANDING_HPO_SEED,
            agent_cwd=agent_cwd,
            fail_closed=True,
            history=LANDING_AGENT_HISTORY,
            explicit_reasoning=True,
            qualitative_notes=True,
        )
        study = oa.create_study(direction="maximize", sampler=sampler, seed=LANDING_HPO_SEED)

        def objective(trial):
            params = suggest_landing_params(trial)
            episodes = _evaluate_landing_params(params, eval_seeds)
            landed = sum(episode["landed"] for episode in episodes)
            mean_return = sum(episode["return"] for episode in episodes) / len(episodes)
            best_episode = max(episodes, key=lambda episode: episode["return"])
            records.append({
                "trial": trial.number,
                "params": params,
                "landing_count": landed,
                "mean_return": mean_return,
                "best_return": best_episode["return"],
                "best_eval_seed": best_episode["seed"],
            })
            return landed * 10000.0 + mean_return

        for _ in range(max_trials):
            study.optimize(objective, n_trials=1)
            current = records[-1]
            print(
                f"trial={current['trial']} landings={current['landing_count']}/"
                f"{LANDING_EVAL_EPISODES} mean_return={current['mean_return']:.3f}"
            )
            if current["landing_count"]:
                break

    best = max(
        records,
        key=lambda record: (
            record["landing_count"], record["best_return"], record["mean_return"]
        ),
    )
    payload = {
        "schema_version": 1,
        "protocol": "lunarlander-animation-controller-v1",
        "method": MODEL_LABEL,
        "backend": BACKEND,
        "model": MODEL,
        "agent_effort": AGENT_EFFORT,
        "history": LANDING_AGENT_HISTORY,
        "explicit_reasoning": True,
        "qualitative_notes": True,
        "hpo_seed": LANDING_HPO_SEED,
        "eval_seeds": list(eval_seeds),
        "max_trials": max_trials,
        "completed_trials": len(records),
        "best_landing_count": best["landing_count"],
        "best_trial": best["trial"],
        "selected_eval_seed": best["best_eval_seed"],
        "selected_return": best["best_return"],
        "selected_params": best["params"],
        "trials": records,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(LANDING_ARTIFACT, payload)
    print(f"wrote {LANDING_ARTIFACT}")
    if not best["landing_count"]:
        raise RuntimeError(f"no successful landing in {len(records)} trials")
    render_landing_gif(payload)
    return payload


def render_landing_gif(payload=None):
    import imageio.v2 as imageio

    gym, _ = _gymnasium()
    if payload is None:
        payload = json.loads(LANDING_ARTIFACT.read_text())
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    state, _ = env.reset(seed=payload["selected_eval_seed"])
    frames = []
    total = 0.0
    final_reward = None
    terminated = truncated = False
    while not (terminated or truncated):
        frames.append(env.render())
        state, reward, terminated, truncated, _ = env.step(
            _landing_action(state, payload["selected_params"])
        )
        total += float(reward)
        final_reward = float(reward)
    frames.append(env.render())
    env.close()
    if not _is_successful_landing(terminated, truncated, final_reward):
        raise RuntimeError("selected LunarLander rollout no longer lands")
    output = ASSETS / "lunarlander_policy.gif"
    imageio.mimsave(output, frames, fps=30)
    print(
        f"wrote {output} (trial={payload['best_trial']}, "
        f"eval_seed={payload['selected_eval_seed']}, return={total:.6f})"
    )
    return output


def landing_metric():
    if not LANDING_ARTIFACT.exists():
        print(0)
        return
    print(json.loads(LANDING_ARTIFACT.read_text()).get("best_landing_count", 0))


def run_one(method, seed, timeout, workers):
    started = time.monotonic()
    values, params = _run_search(method, seed, timeout, workers)
    payload = _common_metadata(method, seed)
    best_values = {}
    best_params = {}
    for env_name in ENVS:
        best_index = max(range(N_TRIALS), key=values[env_name].__getitem__)
        best_values[env_name] = values[env_name][best_index]
        best_params[env_name] = params[best_index]
    payload.update({
        "values": values,
        "params": params,
        "best_values": best_values,
        "best_params": best_params,
        "elapsed_seconds": time.monotonic() - started,
    })
    _validate_artifact(payload, method, seed)
    _atomic_json(_artifact_path(method, seed), payload)
    print(f"wrote {_artifact_path(method, seed)}")


def run_methods(methods, seeds, timeout, workers):
    for method in methods:
        seed_workers = min(workers, len(seeds))
        study_workers = max(1, workers // seed_workers)
        print(f"== {method}: seed_workers={seed_workers}, study_workers={study_workers} ==")
        with ThreadPoolExecutor(max_workers=seed_workers) as pool:
            futures = [
                pool.submit(run_one, method, seed, timeout, study_workers)
                for seed in seeds
            ]
            for future in futures:
                future.result()


def _load_artifact(method, seed):
    path = _artifact_path(method, seed)
    if not path.exists():
        raise FileNotFoundError(f"missing RL control artifact: {path}")
    return _validate_artifact(json.loads(path.read_text()), method, seed)


def _method_summary(method):
    import numpy as np
    runs = [_load_artifact(method, seed) for seed in SEEDS]
    return {
        env_name: {
            "curve": np.asarray([
                _incumbent(run["values"][env_name]) for run in runs
            ]).mean(axis=0),
            "final": float(np.mean([run["best_values"][env_name] for run in runs])),
        }
        for env_name in ENVS
    }


def summary():
    print("method\tAcrobot-v1\tLunarLander-v3")
    for method in METHODS:
        result = _method_summary(method)
        print(
            f"{method}\t{result['Acrobot-v1']['final']:.3f}\t"
            f"{result['LunarLander-v3']['final']:.3f}"
        )


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    styles = {
        "Random": dict(color="#7A7A7A", linestyle=(0, (2, 2)), lw=2.4),
        "TPE": dict(color="#111827", linestyle=(0, (6, 2)), lw=2.4),
        MODEL_LABEL: dict(color="#009E73", marker="o", lw=1.8),
        f"{MODEL_LABEL}-no-context": dict(
            color="#D55E00", marker="D", linestyle=(0, (4, 2)), lw=1.8,
        ),
    }
    display = {
        "Random": "Random",
        "TPE": "TPE",
        MODEL_LABEL: MODEL_LABEL,
        f"{MODEL_LABEL}-no-context": f"{MODEL_LABEL} w/o context",
    }
    fig, axes = plt.subplots(1, len(ENVS), figsize=(11, 4.4))
    trials = range(1, N_TRIALS + 1)
    for ax, env_name in zip(axes, ENVS):
        for method in METHODS:
            ax.plot(
                trials,
                _method_summary(method)[env_name]["curve"],
                label=display[method],
                **styles[method],
            )
        ax.set_title(env_name)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Best mean eval return so far")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.18)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"CPU Gymnasium RL control HPO (5 seeds, {N_TRIALS} trials)")
    fig.tight_layout()
    output = ASSETS / "rl_control.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def _best_lunarlander_case():
    runs = [_load_artifact(MODEL_LABEL, seed) for seed in SEEDS]
    run = max(
        runs,
        key=lambda item: item["best_values"]["LunarLander-v3"],
    )
    values = run["values"]["LunarLander-v3"]
    trial = max(range(N_TRIALS), key=values.__getitem__)
    return run, trial, run["seed"] + trial


def gif():
    if LANDING_ARTIFACT.exists():
        render_landing_gif()
        return
    try:
        import imageio.v2 as imageio
        gym, np = _gymnasium()
    except Exception as error:
        print(f"skipping LunarLander GIF: {error}")
        return
    try:
        run, trial, training_seed = _best_lunarlander_case()
        params = run["params"][trial]
        replayed_mean, q = _evaluate_env("LunarLander-v3", params, training_seed)
        expected_mean = run["values"]["LunarLander-v3"][trial]
        if not math.isclose(replayed_mean, expected_mean, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"LunarLander replay drift: expected {expected_mean}, "
                f"got {replayed_mean}"
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
                state, signal, terminated, truncated, _ = env.step(action)
                total += float(signal)
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
    except Exception as error:
        print(f"skipping LunarLander GIF: {error}")


def selfcheck():
    assert _incumbent([1, 0, 2]) == [1, 1, 2]
    params = [{
        "learning_rate": 0.2,
        "gamma": 0.98,
        "epsilon_decay": 0.96,
        "min_epsilon": 0.05,
        "bins": 8,
        "train_episodes": 16,
    } for _ in range(N_TRIALS)]
    run = _common_metadata(MODEL_LABEL, 0)
    run.update({
        "values": {env_name: [float(i) for i in range(N_TRIALS)] for env_name in ENVS},
        "params": params,
        "best_values": {env_name: float(N_TRIALS - 1) for env_name in ENVS},
        "best_params": {env_name: params[-1] for env_name in ENVS},
        "elapsed_seconds": 1.0,
    })
    _validate_artifact(run, MODEL_LABEL, 0)
    print("selfcheck ok")


def preflight(timeout):
    prompt = 'Reply with ONLY this JSON object: {"ok": true}'
    with tempfile.TemporaryDirectory(prefix="optim-agent-rl-preflight-") as agent_cwd:
        reply = agent_api.call_agent(
            BACKEND, MODEL, prompt, timeout, effort=AGENT_EFFORT, cwd=agent_cwd,
        )
    data = agent_api.extract_json(reply)
    if not data or data.get("ok") is not True:
        raise RuntimeError(f"preflight failed for {MODEL}: invalid JSON reply")
    print(f"preflight ok: {MODEL} ({BACKEND}, {AGENT_EFFORT} effort, history={AGENT_HISTORY})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--method", nargs="+", choices=METHODS, default=list(METHODS))
    run_parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    run_parser.add_argument("--timeout", type=float, default=600)
    run_parser.add_argument("--workers", type=int, default=3)
    landing = sub.add_parser("landing-run")
    landing.add_argument("--max-trials", type=int, default=LANDING_MAX_TRIALS)
    landing.add_argument("--timeout", type=float, default=600)
    check = sub.add_parser("preflight")
    check.add_argument("--timeout", type=float, default=120)
    sub.add_parser("selfcheck")
    sub.add_parser("summary")
    sub.add_parser("plot")
    sub.add_parser("gif")
    sub.add_parser("landing-metric")
    args = parser.parse_args()

    if args.command == "run":
        run_methods(args.method, args.seeds, args.timeout, args.workers)
    elif args.command == "landing-run":
        run_landing_hpo(args.max_trials, args.timeout)
    elif args.command == "preflight":
        preflight(args.timeout)
    elif args.command == "selfcheck":
        selfcheck()
    elif args.command == "summary":
        summary()
    elif args.command == "plot":
        plot()
    elif args.command == "gif":
        gif()
    elif args.command == "landing-metric":
        landing_metric()


if __name__ == "__main__":
    main()
