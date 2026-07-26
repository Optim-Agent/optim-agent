"""GP-BO (skopt EI, Matern-5/2) on the analytic Branin/Ackley-5D suite.

Mirrors examples/hard_functions.py: same functions, bounds, 10 trials,
seeds 0-4, three-point startup. Writes docs/assets/hard_curves_GP-BO_s{seed}.json
in the same schema as the other hard-function artifacts.
"""
import json
import sys

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from hard_functions import FUNCTIONS  # noqa: E402
from skopt import Optimizer  # noqa: E402
from skopt.space import Real  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import ConstantKernel, Matern  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
N_TRIALS = 10
N_INIT = 3


def make_gpr(seed):
    # ponytail: vanilla sklearn GPR; skopt's own GPR subclass segfaults with sklearn 1.7
    return GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(nu=2.5),
        normalize_y=True, alpha=1e-6, n_restarts_optimizer=2, random_state=seed,
    )


def run(fn, bounds, seed):
    dims = [Real(lo, hi) for lo, hi in bounds]
    opt = Optimizer(dims, base_estimator=make_gpr(seed), acq_func="EI",
                    acq_optimizer="sampling", n_initial_points=N_INIT,
                    random_state=seed)
    values, params = [], []
    for _ in range(N_TRIALS):
        x = opt.ask()
        point = [float(v) for v in x]
        y = fn(point)
        opt.tell(x, y)
        values.append(float(y))
        params.append(point)
    return values, params


def main():
    for seed in SEEDS:
        payload = {
            "label": "GP-BO",
            "backend": None,
            "model": None,
            "effort": None,
            "sampler": ("skopt.Optimizer(base_estimator=sklearn "
                        "GaussianProcessRegressor(Matern nu=2.5, normalize_y, "
                        "alpha=1e-6), acq_func='EI', acq_optimizer='sampling', "
                        "n_initial_points=3)"),
            "no_context": True,
            "seed": seed,
            "trials": N_TRIALS,
            "functions": {},
        }
        for name, spec in FUNCTIONS.items():
            vals, params = run(spec["fn"], spec["bounds"], seed)
            payload["functions"][name] = {"values": vals, "params": params}
            print(f"seed {seed} {name}: best={min(vals):.4f}", flush=True)
        out = str(ROOT / "docs" / "assets" / f"hard_curves_GP-BO_s{seed}.json")
        with open(out, "w") as fh:
            json.dump(payload, fh, indent=1)


if __name__ == "__main__":
    main()
