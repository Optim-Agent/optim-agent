"""Offline self-check: `python tests/test_optim_agent.py` (also pytest-compatible)."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optim_agent as oa
from optim_agent import agent, samplers


def quadratic(trial):
    x = trial.suggest_float("x", -5, 5, context="knob that centers a parabola at x=2")
    return (x - 2) ** 2


def test_random_study():
    study = oa.create_study(seed=0)
    study.optimize(quadratic, n_trials=30, verbose=False)
    assert len(study.trials) == 30
    assert all(-5 <= t.params["x"] <= 5 for t in study.trials)
    assert study.best_value < 1.0


def test_extract_json():
    assert agent.extract_json('```json\n{"x": 1.5}\n```') == {"x": 1.5}
    assert agent.extract_json('Sure! Here you go: {"x": 2, "y": "a"} done') == {"x": 2, "y": "a"}
    assert agent.extract_json('schema {"x": "<float>"} answer {"x": 3.0}') == {"x": 3.0}
    assert agent.extract_json("no json here") is None


def test_agent_sampler(monkeypatch=None):
    calls = []

    def fake_call(backend, model, prompt, timeout, effort=None):
        calls.append((prompt, effort))
        return 'reasoning... ```json\n{"x": 2.01, "_note": "min near x=2"}\n```'

    original = samplers._agent.call_agent
    samplers._agent.call_agent = fake_call
    try:
        s = oa.AgentSampler(backend="claude", effort="max", n_init=2, seed=1)
        study = oa.create_study(sampler=s, seed=1)
        study.optimize(quadratic, n_trials=5, verbose=False)
    finally:
        samplers._agent.call_agent = original
    assert calls, "agent was never consulted"
    assert "Search space" in calls[0][0] and "Trial history" in calls[0][0]
    assert "centers a parabola" in calls[0][0], "per-param context not shown to agent"
    assert "min near x=2" in calls[-1][0], "note not carried forward"
    assert calls[0][1] == "max", "effort not forwarded to the CLI call"
    assert abs(study.best_trial.params["x"] - 2.01) < 1e-9
    # effort maps to each CLI's reasoning-effort flag
    assert agent._cmd("claude", None, "p", "high")[-3:-1] == ["--effort", "high"]
    assert "model_reasoning_effort=xhigh" in agent._cmd("codex", None, "p", "xhigh")
    assert "--variant" in agent._cmd("opencode", None, "p", "low")
    assert "--effort" not in agent._cmd("claude", None, "p")  # None effort adds no flag
    # out-of-range and garbage replies fall back gracefully
    samplers._agent.call_agent = lambda *a, **k: '{"x": 999}'
    try:
        study.optimize(quadratic, n_trials=1, verbose=False)  # clamped to 5
        assert study.trials[-1].params["x"] == 5
    finally:
        samplers._agent.call_agent = original


def test_pruner():
    def fake_call(backend, model, prompt, timeout):
        return '{"prune": true}'

    original = agent.call_agent
    from optim_agent import pruners
    pruners._agent.call_agent = fake_call
    try:
        study = oa.create_study(pruner=oa.AgentPruner(level="tight"), seed=0)

        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            for step in range(10):
                trial.report(x + step, step)
                if trial.should_prune():
                    raise oa.TrialPruned()
            return x

        study.optimize(objective, n_trials=3, verbose=False)
        states = [t.state for t in study.trials]
        assert states[0] == "complete"          # nothing to compare against yet
        assert "pruned" in states[1:]           # agent said prune
        pruned = next(t for t in study.trials if t.state == "pruned")
        assert pruned.value == pruned.intermediate[-1][1]
    finally:
        pruners._agent.call_agent = original


def test_mock_backend_and_storage():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "study.json"
        s = oa.AgentSampler(backend="mock", n_init=2, seed=3)
        study = oa.create_study(sampler=s, storage=path, seed=3)
        study.optimize(quadratic, n_trials=15, verbose=False)
        assert study.best_value < 0.5
        resumed = oa.create_study(storage=path, seed=4)
        assert len(resumed.trials) == 15
        assert resumed.space["x"].high == 5
        resumed.optimize(quadratic, n_trials=1, verbose=False)
        assert len(resumed.trials) == 16
        assert json.loads(path.read_text())["trials"][-1]["number"] == 15


def test_concurrency_and_sqlite():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "study.db"
        s = oa.AgentSampler(backend="mock", n_init=2, seed=5)
        study = oa.create_study(sampler=s, storage=path, seed=5, max_concurrency=4)
        study.optimize(quadratic, n_trials=20, verbose=False)
        assert len(study.trials) == 20
        assert all(-5 <= t.params["x"] <= 5 for t in study.trials)
        assert study.best_value < 1.0
        # a second process sharing the same DB sees all history, then extends it
        resumed = oa.create_study(storage=path, seed=6)
        assert len(resumed.trials) == 20
        resumed.optimize(quadratic, n_trials=1, verbose=False)
        nums = [t.number for t in oa.create_study(storage=path).trials]
        assert len(nums) == len(set(nums)) == 21  # AUTOINCREMENT keeps numbers unique


def test_skill_mode_ask_tell():
    study = oa.create_study()
    trial = study.ask({"lr": 3e-4, "batch": 64})  # session agent picks the point
    assert trial.params == {"lr": 3e-4, "batch": 64}
    study.tell(trial, 0.42)
    assert study.best_value == 0.42


def _raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    return False


def test_hostile_agent_values():
    from optim_agent.space import Categorical, Float, Int
    f = Float(0.0, 1.0)
    assert _raises(ValueError, lambda: f.validate(float("nan")))
    assert _raises(ValueError, lambda: f.validate("garbage"))
    assert _raises((ValueError, OverflowError), lambda: Int(1, 10).validate(float("inf")))
    assert Categorical((1, "a")).validate(1.0) == 1  # canonical choice, agent's spelling dropped
    assert type(Categorical((1, "a")).validate(1.0)) is int
    assert _raises(ValueError, lambda: Float(0.0, 1.0, log=True))
    assert _raises(ValueError, lambda: Float(2.0, 1.0))


def test_guardrails():
    # changed bounds mid-study fail loudly
    study = oa.create_study(seed=0)
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1, verbose=False)
    t = study.ask()
    assert _raises(ValueError, lambda: t.suggest_float("x", 0, 2))
    # bad state / valueless complete fail loudly
    assert _raises(ValueError, lambda: study.tell(t, 1.0, state="finished"))
    assert _raises(ValueError, lambda: study.tell(study.ask()))
    # explicit ask(params) is validated once the space is known
    t2 = study.ask({"x": 999})
    assert t2.suggest_float("x", 0, 1) == 1  # clamped
    # numpy scalars unwrap so storage round-trips faithfully
    try:
        import numpy as np
        assert type(oa.create_study().ask({"lr": np.float32(0.1)}).params["lr"]) is float
    except ImportError:
        pass


def test_resume_no_replay():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.json"
        study = oa.create_study(storage=path, seed=7)
        study.optimize(quadratic, n_trials=3, verbose=False)
        first = [t.params["x"] for t in study.trials]
        resumed = oa.create_study(storage=path, seed=7)  # same seed: must not replay
        resumed.optimize(quadratic, n_trials=3, verbose=False)
        fresh = [t.params["x"] for t in resumed.trials[3:]]
        assert not set(first) & set(fresh)
        assert [t.number for t in resumed.trials] == list(range(6))
        assert _raises(ValueError, lambda: oa.create_study(direction="maximize", storage=path))


def test_mnist_helper_curves_and_labels():
    from examples import mnist

    assert mnist._sanitize_label("GPT-5.5") == "GPT-5.5"
    assert mnist._sanitize_label("agent/mock") == "agent-mock"
    assert mnist._best_error_curve([{"test_error": 9.0}, {"test_error": 7.5},
                                    {"test_error": 8.0}]) == [9.0, 7.5, 7.5]
    assert mnist._device_for_trial(10, [0, 1, 2]) == "cuda:1"
    assert mnist._device_for_trial(3, []) == "cpu"


def test_mnist_trial_record_serializes_metrics():
    from examples import mnist

    study = oa.create_study()
    trial = study.ask({"lr": 0.001, "batch_size": 128, "dropout": 0.2, "width": 32})
    trial.suggest_float("lr", 1e-4, 3e-2, log=True)
    trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    trial.suggest_float("dropout", 0.0, 0.6)
    trial.suggest_categorical("width", [16, 32, 64, 96, 128])
    metrics = {"test_error": 2.5, "test_acc": 97.5, "test_loss": 0.08,
               "history": [{"epoch": 1, "test_error": 2.5}]}

    rec = mnist._trial_record(trial, metrics)

    assert rec["params"]["batch_size"] == 128
    assert rec["test_error"] == 2.5
    assert rec["history"][0]["epoch"] == 1


if __name__ == "__main__":
    for fn in [test_random_study, test_extract_json, test_agent_sampler,
               test_pruner, test_mock_backend_and_storage, test_concurrency_and_sqlite,
               test_skill_mode_ask_tell, test_hostile_agent_values, test_guardrails,
               test_resume_no_replay, test_mnist_helper_curves_and_labels,
               test_mnist_trial_record_serializes_metrics]:
        fn()
        print(f"ok: {fn.__name__}")
    print("all checks passed")
