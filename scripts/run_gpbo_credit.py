"""GP-BO (scikit-optimize) baseline on the UCI credit-default task.

Mirrors examples/credit_card.py exactly: same dataset archive, split seed,
search space, 20 trials, seeds 0-4, incumbent selection, and one held-out test
evaluation of the selected incumbent. Writes artifacts in the same schema to
docs/assets/credit_default_GP-BO_s{seed}.json.
"""
import json
import sys
import time
from datetime import datetime, timezone

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import credit_card as cc  # noqa: E402
from skopt import Optimizer  # noqa: E402
from skopt.space import Categorical, Integer, Real  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import ConstantKernel, Matern  # noqa: E402


def make_gpr(seed):
    # ponytail: vanilla sklearn GPR; skopt's own GPR subclass segfaults with sklearn 1.7
    return GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(nu=2.5),
        normalize_y=True, alpha=1e-6, n_restarts_optimizer=2, random_state=seed,
    )

SEEDS = (0, 1, 2, 3, 4)
NONE_TOKEN = "none"

DIMENSIONS = [
    Real(0.01, 0.3, prior="log-uniform", name="learning_rate"),
    Integer(50, 400, name="max_iter"),
    Integer(7, 63, name="max_leaf_nodes"),
    Categorical([NONE_TOKEN, "3", "5", "8"], name="max_depth"),
    Integer(10, 200, prior="log-uniform", name="min_samples_leaf"),
    Real(1e-8, 10.0, prior="log-uniform", name="l2_regularization"),
    Categorical([32, 64, 128, 255], name="max_bins"),
    Real(1.0, 5.0, prior="log-uniform", name="positive_class_weight"),
]
NAMES = [d.name for d in DIMENSIONS]


def decode(point):
    params = dict(zip(NAMES, point))
    params["max_depth"] = None if params["max_depth"] == NONE_TOKEN else int(params["max_depth"])
    params["learning_rate"] = float(params["learning_rate"])
    params["l2_regularization"] = float(params["l2_regularization"])
    params["positive_class_weight"] = float(params["positive_class_weight"])
    params["max_iter"] = int(params["max_iter"])
    params["max_leaf_nodes"] = int(params["max_leaf_nodes"])
    params["min_samples_leaf"] = int(params["min_samples_leaf"])
    params["max_bins"] = int(params["max_bins"])
    return params


def main():
    split, categorical_features, prevalence = cc._load_dataset()
    default = cc._default_reference(split, categorical_features)
    template = json.load(open(
        str(ROOT / "docs" / "assets" / "credit_default_Random_s0.json")))

    for seed in SEEDS:
        started = time.monotonic()
        opt = Optimizer(DIMENSIONS, base_estimator=make_gpr(seed), acq_func="EI",
                        acq_optimizer="sampling",
                        n_initial_points=cc.N_INIT, random_state=seed)
        values, params_list = [], []
        for _ in range(cc.N_TRIALS):
            point = opt.ask()
            params = decode(point)
            value = cc._evaluate_params(params, split, categorical_features)[
                "validation_log_loss"]
            opt.tell(point, value)
            values.append(float(value))
            params_list.append(params)
        best_index = min(range(cc.N_TRIALS), key=values.__getitem__)
        held_out = cc._evaluate_params(
            params_list[best_index], split, categorical_features, include_test=True)

        payload = dict(template)
        payload.update({
            "method": "GP-BO",
            "backend": None,
            "model": None,
            "effort": None,
            "use_context": False,
            "context_policy": "no context (numerical GP-BO baseline)",
            "task_context": None,
            "agent_failure_policy": None,
            "n_init": cc.N_INIT,
            "seed": seed,
            "sampler": ("skopt.Optimizer(base_estimator=sklearn GaussianProcessRegressor"
                        "(Matern nu=2.5, normalize_y, alpha=1e-6), acq_func='EI')"),
            "skopt_version": __import__("skopt").__version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "default_prevalence": prevalence,
            "values": values,
            "params": params_list,
            "best_validation_log_loss": values[best_index],
            "best_params": params_list[best_index],
            "test_log_loss": held_out["test_log_loss"],
            "default_validation_log_loss": default["validation_log_loss"],
            "default_test_log_loss": default["test_log_loss"],
            "elapsed_seconds": time.monotonic() - started,
            "history": None,
            "explicit_reasoning": None,
            "qualitative_notes": None,
        })
        out = str(ROOT / "docs" / "assets" / f"credit_default_GP-BO_s{seed}.json")
        with open(out, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"seed {seed}: best_val={values[best_index]:.6f} "
              f"test={held_out['test_log_loss']:.6f} "
              f"({time.monotonic() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
