"""CPU-only context benchmark on UCI credit-default data.

    python examples/credit_card.py download
    python examples/credit_card.py preflight
    python examples/credit_card.py run
    python examples/credit_card.py selfcheck
    python examples/credit_card.py summary
    python examples/credit_card.py plot
"""

import argparse
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optim_agent as oa
from optim_agent import agent as agent_api


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
DATA_DIR = ROOT / "data" / "uci-default-credit"
DATASET_ID = 350
DATASET_PAGE = (
    "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
)
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)
DATA_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
ARCHIVE_MEMBER = "default of credit card clients.xls"
DATA_LICENSE = "CC BY 4.0"
DATA_DOI = "10.24432/C55S3H"
DATA_ARCHIVE = DATA_DIR / "default-of-credit-card-clients.zip"
DEFAULT_PREVALENCE = 6636 / 30000

BACKEND = "codex"
MODEL = "gpt-5.5"
MODEL_LABEL = "GPT-5.5"
SEEDS = (0, 1, 2, 3, 4)
N_TRIALS = 20
N_INIT = 3
SPLIT_SEED = 20260713
SEARCH_SPACE_VERSION = "uci-credit-default-hgb-v1"
PROTOCOL_VERSION = "credit-default-context-v2"
BASELINE_PROTOCOL_VERSION = "credit-default-context-effort-v1"
AGENT_EFFORT = "high"
HISTORY = 20
EXPLICIT_REASONING = True
QUALITATIVE_NOTES = True
NO_CONTEXT_METHOD = f"{MODEL_LABEL}-no-context"
METHODS = (
    "Random",
    "TPE",
    "GP-BO",
    MODEL_LABEL,
    NO_CONTEXT_METHOD,
)
SELECTED_METHOD = MODEL_LABEL
TASK_CONTEXT = (
    "Minimize validation log loss for probability of next-month default on the UCI "
    "Default of Credit Card Clients dataset. It has 30,000 rows, 23 demographic, "
    "credit-limit, bill, payment, and six-month repayment-status features, with the "
    "default class in the minority. Tune a CPU HistGradientBoostingClassifier within "
    "a 20-trial budget. Favor calibrated probabilities and regularized interactions; "
    "more leaves, depth, bins, iterations, or class weighting can improve fit but may "
    "overfit. The held-out test set is not visible during optimization."
)

TARGET_COLUMN = "default payment next month"
PAY_STATUS_COLUMNS = ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
CATEGORICAL_COLUMNS = ("SEX", "EDUCATION", "MARRIAGE", *PAY_STATUS_COLUMNS)
RAW_COLUMNS = (
    "ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    *PAY_STATUS_COLUMNS,
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5",
    "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5",
    "PAY_AMT6", TARGET_COLUMN,
)

PARAMETER_CONTEXT = {
    "learning_rate": "boosting shrinkage; smaller values usually need more iterations",
    "max_iter": "maximum boosting iterations, trading CPU cost for model capacity",
    "max_leaf_nodes": "maximum leaves per tree, controlling nonlinear interaction capacity",
    "max_depth": "tree depth cap; shallower trees regularize interactions",
    "min_samples_leaf": "minimum samples per leaf; larger values smooth noisy credit segments",
    "l2_regularization": "L2 penalty on leaf values, reducing overconfident probabilities",
    "max_bins": "histogram resolution for continuous credit and payment amounts",
    "positive_class_weight": "relative training weight for the minority default class",
}


def _context(name, use_context):
    return PARAMETER_CONTEXT[name] if use_context else None


def suggest_params(trial, use_context=True):
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True,
            context=_context("learning_rate", use_context),
        ),
        "max_iter": trial.suggest_int(
            "max_iter", 50, 400, context=_context("max_iter", use_context),
        ),
        "max_leaf_nodes": trial.suggest_int(
            "max_leaf_nodes", 7, 63, context=_context("max_leaf_nodes", use_context),
        ),
        "max_depth": trial.suggest_categorical(
            "max_depth", (None, 3, 5, 8), context=_context("max_depth", use_context),
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 10, 200, log=True,
            context=_context("min_samples_leaf", use_context),
        ),
        "l2_regularization": trial.suggest_float(
            "l2_regularization", 1e-8, 10.0, log=True,
            context=_context("l2_regularization", use_context),
        ),
        "max_bins": trial.suggest_categorical(
            "max_bins", (32, 64, 128, 255), context=_context("max_bins", use_context),
        ),
        "positive_class_weight": trial.suggest_float(
            "positive_class_weight", 1.0, 5.0, log=True,
            context=_context("positive_class_weight", use_context),
        ),
    }


def _prepare_frame(frame):
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if set(frame.columns) != set(RAW_COLUMNS):
        raise ValueError("incompatible UCI credit-default schema")
    target = frame.pop(TARGET_COLUMN)
    frame = frame.drop(columns=["ID"])
    if set(target.unique()) - {0, 1}:
        raise ValueError("incompatible UCI credit-default target")
    for column in PAY_STATUS_COLUMNS:
        frame[column] = frame[column] + 2
    categorical = [column in CATEGORICAL_COLUMNS for column in frame]
    return frame, target.astype(int), categorical


def _split_data(features, target):
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as error:
        raise SystemExit('Install the ML extra: pip install -e ".[ml]"') from error

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features,
        target,
        test_size=0.4,
        random_state=SPLIT_SEED,
        stratify=target,
    )
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=0.5,
        random_state=SPLIT_SEED,
        stratify=y_holdout,
    )
    return {
        "x_train": x_train,
        "x_valid": x_valid,
        "x_test": x_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
    }


def _build_model(params, categorical_features):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as error:
        raise SystemExit('Install the ML extra: pip install -e ".[ml]"') from error

    model_params = dict(params)
    positive_weight = model_params.pop("positive_class_weight")
    return HistGradientBoostingClassifier(
        **model_params,
        categorical_features=categorical_features,
        class_weight={0: 1.0, 1: positive_weight},
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=SPLIT_SEED,
    )


def _method_spec(method):
    if method == "Random":
        return {
            "backend": None, "model": None, "effort": None,
            "use_context": False, "n_init": 0,
        }
    if method == "TPE":
        return {
            "backend": "tpe", "model": None, "effort": None,
            "use_context": False, "n_init": N_INIT,
        }
    if method == "GP-BO":
        return {
            "backend": None, "model": None, "effort": None,
            "use_context": False, "n_init": N_INIT,
        }
    if method in (MODEL_LABEL, NO_CONTEXT_METHOD):
        return {
            "backend": BACKEND,
            "model": MODEL,
            "effort": AGENT_EFFORT,
            "use_context": method == MODEL_LABEL,
            "n_init": N_INIT,
        }
    raise ValueError(f"unknown credit-default method: {method}")


class OptunaTrialAdapter:
    def __init__(self, trial):
        self.trial = trial

    def suggest_float(self, name, low, high, *, log=False, context=None):
        return self.trial.suggest_float(name, low, high, log=log)

    def suggest_int(self, name, low, high, *, log=False, context=None):
        return self.trial.suggest_int(name, low, high, log=log)

    def suggest_categorical(self, name, choices, *, context=None):
        return self.trial.suggest_categorical(name, choices)


class StaticTrialAdapter:
    def __init__(self, params):
        self.params = params

    def suggest_float(self, name, low, high, *, log=False, context=None):
        return self.params[name]

    def suggest_int(self, name, low, high, *, log=False, context=None):
        return self.params[name]

    def suggest_categorical(self, name, choices, *, context=None):
        return self.params[name]


def _evaluate_params(params, split, categorical_features, include_test=False):
    try:
        from sklearn.metrics import log_loss
    except ImportError as error:
        raise SystemExit('Install the ML extra: pip install -e ".[ml]"') from error
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        limiter = nullcontext()
    else:
        limiter = threadpool_limits(limits=1)

    model = _build_model(params, categorical_features)
    with limiter:
        model.fit(split["x_train"], split["y_train"])
        valid_probability = model.predict_proba(split["x_valid"])[:, 1]
        metrics = {
            "validation_log_loss": float(log_loss(
                split["y_valid"], valid_probability, labels=[0, 1],
            )),
        }
        if include_test:
            test_probability = model.predict_proba(split["x_test"])[:, 1]
            metrics["test_log_loss"] = float(log_loss(
                split["y_test"], test_probability, labels=[0, 1],
            ))
    return metrics


def _make_sampler(method, seed, timeout, agent_cwd, history=HISTORY,
                  explicit_reasoning=EXPLICIT_REASONING,
                  qualitative_notes=QUALITATIVE_NOTES):
    spec = _method_spec(method)
    if method == "Random":
        return oa.RandomSampler()
    if method in ("TPE", "GP-BO"):
        raise ValueError(f"{method} uses its external sampler directly")
    return oa.AgentSampler(
        backend=spec["backend"],
        model=spec["model"],
        effort=spec["effort"],
        context=TASK_CONTEXT if spec["use_context"] else None,
        n_init=spec["n_init"],
        timeout=timeout,
        seed=seed,
        agent_cwd=agent_cwd,
        fail_closed=True,
        history=history,
        explicit_reasoning=explicit_reasoning,
        qualitative_notes=qualitative_notes,
    )


def _artifact_path(method, seed):
    return ASSETS / f"credit_default_{method}_s{seed}.json"


def _validate_archive(path, expected_hash=DATA_SHA256, member=ARCHIVE_MEMBER):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"credit-default archive SHA-256 mismatch: {digest}")
    try:
        with zipfile.ZipFile(path) as bundle:
            names = bundle.namelist()
    except zipfile.BadZipFile as error:
        raise ValueError("invalid credit-default archive") from error
    if names != [member]:
        raise ValueError(f"unexpected credit-default archive member: {names}")
    return path


def _context_policy(method):
    spec = _method_spec(method)
    if method == "GP-BO":
        return "no context (numerical GP-BO baseline)"
    if spec["backend"] in (None, "tpe"):
        return "not applicable"
    return "supplied task and parameter context" if spec["use_context"] else (
        "no supplied task context"
    )


def _common_metadata(method, seed, history=HISTORY,
                     explicit_reasoning=EXPLICIT_REASONING,
                     qualitative_notes=QUALITATIVE_NOTES):
    spec = _method_spec(method)
    try:
        import sklearn
    except ImportError:
        sklearn_version = None
    else:
        sklearn_version = sklearn.__version__
    return {
        "schema_version": 1,
        "protocol": (
            PROTOCOL_VERSION if spec["backend"] == BACKEND
            else BASELINE_PROTOCOL_VERSION
        ),
        "method": method,
        "backend": spec["backend"],
        "model": spec["model"],
        "effort": spec["effort"],
        "use_context": (
            spec["use_context"] if spec["backend"] == BACKEND or method == "GP-BO" else None
        ),
        "context_policy": _context_policy(method),
        "task_context": TASK_CONTEXT if spec["backend"] == BACKEND and spec["use_context"] else None,
        "agent_failure_policy": (
            "fail_closed" if spec["backend"] == BACKEND
            else None if method == "GP-BO"
            else "not applicable"
        ),
        "n_init": spec["n_init"],
        "history": history if spec["backend"] == BACKEND else None,
        "explicit_reasoning": explicit_reasoning if spec["backend"] == BACKEND else None,
        "qualitative_notes": qualitative_notes if spec["backend"] == BACKEND else None,
        "seed": seed,
        "trials": N_TRIALS,
        "split_seed": SPLIT_SEED,
        "search_space_version": SEARCH_SPACE_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_url": DATA_URL,
        "dataset_sha256": DATA_SHA256,
        "dataset_license": DATA_LICENSE,
        "dataset_doi": DATA_DOI,
        "default_prevalence": DEFAULT_PREVALENCE,
        "objective": "validation log loss",
        "sklearn_version": sklearn_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _valid_number(value, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    return (low is None or value >= low) and (high is None or value <= high)


def _valid_params(params):
    expected = set(PARAMETER_CONTEXT)
    if not isinstance(params, list) or len(params) != N_TRIALS:
        return False
    for point in params:
        if not isinstance(point, dict) or set(point) != expected:
            return False
        if not _valid_number(point["learning_rate"], 0.01, 0.3):
            return False
        if not _valid_number(point["max_iter"], 50, 400):
            return False
        if not _valid_number(point["max_leaf_nodes"], 7, 63):
            return False
        if point["max_depth"] not in (None, 3, 5, 8):
            return False
        if not _valid_number(point["min_samples_leaf"], 10, 200):
            return False
        if not _valid_number(point["l2_regularization"], 1e-8, 10.0):
            return False
        if point["max_bins"] not in (32, 64, 128, 255):
            return False
        if not _valid_number(point["positive_class_weight"], 1.0, 5.0):
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
    expected.pop("sklearn_version")
    valid = all(run.get(key) == value for key, value in expected.items())
    valid = valid and isinstance(run.get("sklearn_version"), str)
    valid = valid and _has_valid_creation_time(run)
    values = run.get("values")
    valid = valid and isinstance(values, list) and len(values) == N_TRIALS
    valid = valid and all(_valid_number(value, 0.0) for value in values or ())
    valid = valid and _valid_params(run.get("params"))
    if valid:
        best_index = min(range(N_TRIALS), key=values.__getitem__)
        best_value = run.get("best_validation_log_loss")
        valid = _valid_number(best_value, 0.0)
        valid = valid and math.isclose(
            best_value, values[best_index], rel_tol=1e-12, abs_tol=1e-12,
        )
        valid = valid and run.get("best_params") == run["params"][best_index]
    valid = valid and _valid_number(run.get("test_log_loss"), 0.0)
    valid = valid and _valid_number(run.get("default_validation_log_loss"), 0.0)
    valid = valid and _valid_number(run.get("default_test_log_loss"), 0.0)
    valid = valid and _valid_number(run.get("elapsed_seconds"), 0.0)
    if not valid:
        raise ValueError(f"incompatible credit-default artifact for {method} seed {seed}")
    return run


def _run_search(method, seed, split, categorical_features, timeout, agent_cwd=None,
                history=HISTORY, explicit_reasoning=EXPLICIT_REASONING,
                qualitative_notes=QUALITATIVE_NOTES):
    spec = _method_spec(method)

    def objective(trial):
        params = suggest_params(trial, use_context=spec["use_context"])
        return _evaluate_params(params, split, categorical_features)["validation_log_loss"]

    if method == "TPE":
        try:
            import optuna
        except ImportError as error:
            raise SystemExit('Install examples: pip install -e ".[examples]"') from error
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=N_INIT),
        )
        study.optimize(
            lambda trial: objective(OptunaTrialAdapter(trial)),
            n_trials=N_TRIALS,
        )
        return [trial.value for trial in study.trials], [trial.params for trial in study.trials]

    if method == "GP-BO":
        try:
            import skopt
            from skopt.space import Categorical, Integer, Real
        except ImportError as error:
            raise SystemExit('Install examples: pip install -e ".[examples]"') from error
        dimensions = [
            Real(0.01, 0.3, prior="log-uniform", name="learning_rate"),
            Integer(50, 400, name="max_iter"),
            Integer(7, 63, name="max_leaf_nodes"),
            Categorical([None, 3, 5, 8], name="max_depth"),
            Integer(10, 200, prior="log-uniform", name="min_samples_leaf"),
            Real(1e-8, 10.0, prior="log-uniform", name="l2_regularization"),
            Categorical([32, 64, 128, 255], name="max_bins"),
            Real(1.0, 5.0, prior="log-uniform", name="positive_class_weight"),
        ]
        optimizer = skopt.Optimizer(
            dimensions,
            base_estimator="GP",
            acq_func="EI",
            n_initial_points=N_INIT,
            random_state=seed,
        )
        values, params = [], []
        names = [dimension.name for dimension in dimensions]
        for _ in range(N_TRIALS):
            point = optimizer.ask()
            candidate = dict(zip(names, point))
            candidate["max_iter"] = int(candidate["max_iter"])
            candidate["max_leaf_nodes"] = int(candidate["max_leaf_nodes"])
            candidate["min_samples_leaf"] = int(candidate["min_samples_leaf"])
            value = objective(StaticTrialAdapter(candidate))
            optimizer.tell(point, value)
            values.append(value)
            params.append(candidate)
        return values, params

    sampler = _make_sampler(
        method, seed, timeout, agent_cwd, history,
        explicit_reasoning, qualitative_notes,
    )
    study = oa.create_study(direction="minimize", sampler=sampler, seed=seed)
    study.optimize(objective, n_trials=N_TRIALS)
    return [trial.value for trial in study.trials], [trial.params for trial in study.trials]


def _ensure_archive(path=DATA_ARCHIVE):
    path = Path(path)
    if path.exists():
        return _validate_archive(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "optim-agent/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        _validate_archive(temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _load_dataset(path=DATA_ARCHIVE):
    try:
        import pandas as pd
    except ImportError as error:
        raise SystemExit('Install the ML extra: pip install -e ".[ml]"') from error

    path = _ensure_archive(path)
    with zipfile.ZipFile(path) as bundle:
        workbook = bundle.read(ARCHIVE_MEMBER)
    frame = pd.read_excel(io.BytesIO(workbook), header=1, engine="xlrd")
    if frame.shape != (30000, 25):
        raise ValueError(f"incompatible UCI credit-default shape: {frame.shape}")
    features, target, categorical = _prepare_frame(frame)
    return _split_data(features, target), categorical, float(target.mean())


def _default_reference(split, categorical_features):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import log_loss
    except ImportError as error:
        raise SystemExit('Install the ML extra: pip install -e ".[ml]"') from error
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        limiter = nullcontext()
    else:
        limiter = threadpool_limits(limits=1)

    model = HistGradientBoostingClassifier(
        categorical_features=categorical_features,
        random_state=SPLIT_SEED,
    )
    with limiter:
        model.fit(split["x_train"], split["y_train"])
        valid_probability = model.predict_proba(split["x_valid"])[:, 1]
        test_probability = model.predict_proba(split["x_test"])[:, 1]
    return {
        "validation_log_loss": float(log_loss(
            split["y_valid"], valid_probability, labels=[0, 1],
        )),
        "test_log_loss": float(log_loss(
            split["y_test"], test_probability, labels=[0, 1],
        )),
    }


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1) + "\n")
    temporary.replace(path)


def run_one(method, seed, timeout, dataset=None, history=HISTORY,
            explicit_reasoning=EXPLICIT_REASONING,
            qualitative_notes=QUALITATIVE_NOTES):
    split, categorical_features, prevalence = dataset or _load_dataset()
    default = _default_reference(split, categorical_features)
    started = time.monotonic()
    spec = _method_spec(method)
    context = (
        tempfile.TemporaryDirectory(prefix="optim-agent-credit-")
        if spec["backend"] == BACKEND else nullcontext(None)
    )
    with context as agent_cwd:
        values, params = _run_search(
            method,
            seed,
            split,
            categorical_features,
            timeout,
            agent_cwd=agent_cwd,
            history=history,
            explicit_reasoning=explicit_reasoning,
            qualitative_notes=qualitative_notes,
        )
    best_index = min(range(N_TRIALS), key=values.__getitem__)
    held_out = _evaluate_params(
        params[best_index], split, categorical_features, include_test=True,
    )
    payload = _common_metadata(
        method, seed, history, explicit_reasoning, qualitative_notes,
    )
    payload.update({
        "default_prevalence": prevalence,
        "values": values,
        "params": params,
        "best_validation_log_loss": values[best_index],
        "best_params": params[best_index],
        "test_log_loss": held_out["test_log_loss"],
        "default_validation_log_loss": default["validation_log_loss"],
        "default_test_log_loss": default["test_log_loss"],
        "elapsed_seconds": time.monotonic() - started,
    })
    _validate_artifact(payload, method, seed)
    _atomic_json(_artifact_path(method, seed), payload)
    print(
        f"wrote {_artifact_path(method, seed)} "
        f"(valid={payload['best_validation_log_loss']:.6f}, "
        f"test={payload['test_log_loss']:.6f})"
    )


def run_methods(methods, seeds, timeout, workers, history=HISTORY,
                explicit_reasoning=EXPLICIT_REASONING,
                qualitative_notes=QUALITATIVE_NOTES):
    dataset = _load_dataset()
    for method in methods:
        print(f"== {method}: seeds {list(seeds)} ==")
        with ThreadPoolExecutor(max_workers=min(workers, len(seeds))) as pool:
            futures = [
                pool.submit(
                    run_one, method, seed, timeout, dataset, history,
                    explicit_reasoning, qualitative_notes,
                )
                for seed in seeds
            ]
            for future in futures:
                future.result()


def _load_artifact(method, seed):
    path = _artifact_path(method, seed)
    if not path.exists():
        raise FileNotFoundError(f"missing credit-default artifact: {path}")
    return _validate_artifact(json.loads(path.read_text()), method, seed)


def _incumbent(values):
    best = math.inf
    curve = []
    for value in values:
        best = min(best, value)
        curve.append(best)
    return curve


def _method_summary(method):
    import numpy as np

    runs = [_load_artifact(method, seed) for seed in SEEDS]
    curves = np.asarray([_incumbent(run["values"]) for run in runs])
    return {
        "curve": curves.mean(axis=0),
        "mean_incumbent": float(curves.mean(axis=1).mean()),
        "final_validation": float(np.mean([run["best_validation_log_loss"] for run in runs])),
        "test_log_loss": float(np.mean([run["test_log_loss"] for run in runs])),
        "default_validation": float(np.mean([
            run["default_validation_log_loss"] for run in runs
        ])),
        "default_test": float(np.mean([run["default_test_log_loss"] for run in runs])),
    }


def selected_summary():
    return {
        "selected_method": SELECTED_METHOD,
        "selected": _method_summary(SELECTED_METHOD),
        "random": _method_summary("Random"),
        "tpe": _method_summary("TPE"),
        "gp_bo": _method_summary("GP-BO"),
    }


def selfcheck():
    _load_dataset()
    assert SELECTED_METHOD in METHODS
    for method in METHODS:
        for seed in SEEDS:
            _load_artifact(method, seed)
    print("selfcheck ok: complete five-seed UCI credit-default artifacts are compatible")


def summary():
    print("method\tmean incumbent\tfinal validation\ttest log loss")
    for method in METHODS:
        result = _method_summary(method)
        print(
            f"{method}\t{result['mean_incumbent']:.6f}\t"
            f"{result['final_validation']:.6f}\t{result['test_log_loss']:.6f}"
        )
    selected = selected_summary()
    print(f"selected\t{selected['selected_method']}")


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    styles = {
        "Random": dict(color="#7A7A7A", linestyle=(0, (2, 2)), lw=2.5),
        "TPE": dict(color="#111827", linestyle=(0, (6, 2)), lw=2.5),
        "GP-BO": dict(color="#0072B2", linestyle=(0, (1, 1)), lw=2.2),
        SELECTED_METHOD: dict(color="#D55E00", marker="^", lw=1.8),
        NO_CONTEXT_METHOD: dict(
            color="#CC79A7", marker="D", linestyle=(0, (4, 2)), lw=1.8,
        ),
    }
    labels = {
        "Random": "Random",
        "TPE": "TPE",
        "GP-BO": "GP-BO",
        SELECTED_METHOD: MODEL_LABEL,
        NO_CONTEXT_METHOD: f"{MODEL_LABEL} w/o context",
    }

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    trials = range(1, N_TRIALS + 1)
    for method in ("Random", "TPE", "GP-BO", SELECTED_METHOD, NO_CONTEXT_METHOD):
        ax.plot(
            trials,
            _method_summary(method)["curve"],
            ms=4.2,
            label=labels[method],
            **styles[method],
        )
    ax.set_title("UCI credit-default HGB tuning (5 seeds, 20 trials)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best validation log loss so far (mean over seeds)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.18)
    ax.legend(fontsize=8.2, ncol=2)
    fig.tight_layout()
    output = ASSETS / "credit_card.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def preflight(timeout):
    prompt = 'Reply with ONLY this JSON object: {"ok": true}'
    with tempfile.TemporaryDirectory(prefix="optim-agent-credit-preflight-") as agent_cwd:
        reply = agent_api.call_agent(
            BACKEND,
            MODEL,
            prompt,
            timeout,
            effort=AGENT_EFFORT,
            cwd=agent_cwd,
        )
    data = agent_api.extract_json(reply)
    if not data or data.get("ok") is not True:
        raise RuntimeError(f"preflight failed for {MODEL}: invalid JSON reply")
    print(f"preflight ok: {MODEL} ({BACKEND}, {AGENT_EFFORT} effort)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--method", nargs="+", choices=METHODS, default=list(METHODS))
    run_parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    run_parser.add_argument("--timeout", type=float, default=600)
    run_parser.add_argument("--workers", type=int, default=5)
    run_parser.add_argument("--history", type=int, default=HISTORY)
    run_parser.add_argument(
        "--explicit-reasoning", action=argparse.BooleanOptionalAction,
        default=EXPLICIT_REASONING,
    )
    run_parser.add_argument(
        "--qualitative-notes", action=argparse.BooleanOptionalAction,
        default=QUALITATIVE_NOTES,
    )
    check = sub.add_parser("preflight")
    check.add_argument("--timeout", type=float, default=120)
    sub.add_parser("selfcheck")
    sub.add_parser("summary")
    sub.add_parser("plot")
    args = parser.parse_args()

    if args.command == "download":
        split, _, prevalence = _load_dataset()
        print(
            f"dataset ok: train={len(split['x_train'])}, valid={len(split['x_valid'])}, "
            f"test={len(split['x_test'])}, default prevalence={prevalence:.4f}"
        )
    elif args.command == "run":
        run_methods(
            args.method, args.seeds, args.timeout, args.workers, args.history,
            args.explicit_reasoning, args.qualitative_notes,
        )
    elif args.command == "preflight":
        preflight(args.timeout)
    elif args.command == "selfcheck":
        selfcheck()
    elif args.command == "summary":
        summary()
    else:
        plot()


if __name__ == "__main__":
    main()
