"""MNIST CNN hyperparameter benchmark.

Full MNIST train/test splits, small CNN, and optim-agent samplers. Independent
trials can run across all visible GPUs; this is trial parallelism, not
distributed training.

    python examples/mnist.py download
    python examples/mnist.py run --method mock --seeds 0 1 2 --workers 8 --gpus 0 1 2 3 4 5 6 7
    python examples/mnist.py run --method Random --seeds 0 1 2 --workers 8 --gpus 0 1 2 3 4 5 6 7
    python examples/mnist.py plot
    python examples/mnist.py selfcheck
"""

import argparse
import json
import math
import os
import random
import re
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optim_agent as oa

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
DATA = ROOT / "data" / "mnist"
STORAGE = ROOT / ".optim-agent-runs" / "mnist"

METHODS = {
    "Random": dict(backend=None, model=None, color="#9ca3af", style="solid"),
    "TPE": dict(backend="tpe", model=None, color="#111827", style=(0, (2, 2))),
    "mock": dict(backend="mock", model=None, color="#2563eb", style=(0, (4, 2))),
    "codex": dict(backend="codex", model=None, label="GPT-5.5", color="#10a37f", style=(0, (6, 2))),
    "claude": dict(backend="claude", model=None, color="#8b5cf6", style="solid"),
    "opencode": dict(backend="opencode", model=None, color="#16a34a", style=(0, (1, 1.5))),
}
BATCHES = [64, 128, 256, 512]
WIDTHS = [16, 32, 64, 96, 128]
PLOT_LABELS = ("Random", "TPE", "GPT-5.5-low", "GPT-5.5-medium", "GPT-5.5-xhigh")
PLOT_STYLES = {
    "GPT-5.5-low": dict(style=(0, (1, 1.5))),
    "GPT-5.5-medium": dict(style=(0, (4, 2))),
    "GPT-5.5-xhigh": dict(style=(0, (6, 2))),
}


def _ensure_cuda_driver_compat():
    """Re-exec with the matching 470 libcuda when the container symlink points at 580.

    The current A800/A100 image exposes `nvidia-smi` via a 470 kernel driver, but
    libcuda.so may resolve to a newer user-mode library. PyTorch then fails with
    CUDA error 803 until the matching library is preloaded.
    """
    if os.environ.get("OPTIM_AGENT_CUDA_COMPAT") == "1":
        return
    lib = Path("/usr/lib/x86_64-linux-gnu/libcuda.so.470.199.02")
    if not lib.exists():
        return
    try:
        import torch
        if torch.cuda.is_available():
            return
    except Exception:
        pass
    env = os.environ.copy()
    env["OPTIM_AGENT_CUDA_COMPAT"] = "1"
    env["LD_PRELOAD"] = str(lib) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


def _sanitize_label(label):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-")


def _run_label(method, effort):
    preset = METHODS[method]
    return f"{preset.get('label', method)}-{effort}" if preset["backend"] not in (None, "tpe") else method


def _device_for_trial(number, gpus):
    return f"cuda:{gpus[number % len(gpus)]}" if gpus else "cpu"


def _best_error_curve(records):
    best, out = math.inf, []
    for rec in records:
        if rec.get("state", "complete") == "complete" and rec.get("test_error") is not None:
            best = min(best, float(rec["test_error"]))
        out.append(best)
    return out


def _trial_record(trial, metrics):
    return {
        "trial": trial.number,
        "state": str(getattr(trial, "state", "complete")).lower(),
        "params": dict(trial.params),
        "test_error": metrics.get("test_error"),
        "test_acc": metrics.get("test_acc"),
        "test_loss": metrics.get("test_loss"),
        "history": metrics.get("history", []),
    }


def _torch():
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except Exception as e:
        raise SystemExit("PyTorch is required for examples/mnist.py") from e
    return torch, nn, F


def _cuda_ready():
    try:
        torch, _, _ = _torch()
        return torch.cuda.is_available()
    except Exception:
        return False


class SmallCNN:
    """Factory wrapper so importing this module does not require torch."""

    @staticmethod
    def make(width, dropout):
        torch, nn, F = _torch()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, width, 3, padding=1)
                self.conv2 = nn.Conv2d(width, width * 2, 3, padding=1)
                self.drop = nn.Dropout(float(dropout))
                self.fc = nn.Linear(width * 2 * 7 * 7, 10)

            def forward(self, x):
                x = F.max_pool2d(F.relu(self.conv1(x)), 2)
                x = F.max_pool2d(F.relu(self.conv2(x)), 2)
                x = self.drop(x.flatten(1))
                return self.fc(x)

        return Net()


def _datasets(download):
    try:
        from torchvision import datasets, transforms
    except Exception as e:
        raise SystemExit("torchvision is required for examples/mnist.py") from e
    try:
        train = datasets.MNIST(str(DATA), train=True, download=download, transform=transforms.ToTensor())
        test = datasets.MNIST(str(DATA), train=False, download=download, transform=transforms.ToTensor())
    except Exception as e:
        if download:
            raise SystemExit("MNIST download failed; please download the full dataset manually "
                             f"into {DATA}") from e
        raise SystemExit(f"MNIST not found in {DATA}; run `python examples/mnist.py download`") from e
    if len(train) != 60000 or len(test) != 10000:
        raise SystemExit(f"expected full MNIST (60000/10000), got {len(train)}/{len(test)}")
    return train, test


def _tensor_dataset(dataset):
    torch, _, _ = _torch()
    x = dataset.data.unsqueeze(1).float().div_(255.0)
    y = dataset.targets.long()
    return torch.utils.data.TensorDataset(x, y)


def _loader(dataset, batch_size, train, seed):
    torch, _, _ = _torch()
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=0,
        pin_memory=_cuda_ready(), generator=g,
    )


def _evaluate(model, loader, device):
    torch, _, F = _torch()
    model.eval()
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    acc = 100.0 * correct / total
    return {"test_loss": loss / total, "test_acc": acc, "test_error": 100.0 - acc}


def _train_once(params, device, epochs, seed):
    torch, _, F = _torch()
    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    train_ds, test_ds = map(_tensor_dataset, _datasets(download=False))
    train_loader = _loader(train_ds, int(params["batch_size"]), True, seed)
    test_loader = _loader(test_ds, 1024, False, seed)
    model = SmallCNN.make(int(params["width"]), float(params["dropout"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]), weight_decay=1e-4)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            opt.step()
        metrics = _evaluate(model, test_loader, device)
        metrics["epoch"] = epoch
        history.append(metrics)
    out = dict(history[-1])
    out["history"] = history
    return out


class _OptunaTrialAdapter:
    def __init__(self, trial):
        self._trial = trial

    @property
    def number(self):
        return self._trial.number

    @property
    def params(self):
        return self._trial.params

    def suggest_float(self, name, low, high, *, context=None, log=False):
        return self._trial.suggest_float(name, low, high, log=log)

    def suggest_categorical(self, name, choices, *, context=None):
        return self._trial.suggest_categorical(name, choices)

    def report(self, value, step):
        return self._trial.report(value, step)


def _objective(epochs, seed, gpus):
    lock = threading.Lock()
    next_slot = 0

    def objective(trial):
        nonlocal next_slot
        with lock:
            slot = next_slot
            next_slot += 1
        lr = trial.suggest_float("lr", 1e-4, 3e-2, log=True,
                                 context="AdamW learning rate for MNIST CNN")
        batch_size = trial.suggest_categorical(
            "batch_size", BATCHES,
            context="mini-batch size; larger improves GPU use but may need a larger learning rate",
        )
        dropout = trial.suggest_float("dropout", 0.0, 0.6,
                                      context="dropout before the classifier head")
        width = trial.suggest_categorical(
            "width", WIDTHS,
            context="base convolution channel count; second conv uses 2x this width",
        )
        device = _device_for_trial(slot, gpus)
        metrics = _train_once(dict(lr=lr, batch_size=batch_size, dropout=dropout, width=width),
                              device, epochs, seed + slot)
        metrics["device"] = device
        for row in metrics["history"]:
            trial.report(row["test_error"], row["epoch"])
        trial._mnist_metrics = metrics
        return metrics["test_error"]
    return objective


def _sampler(method, seed, effort, timeout, model):
    preset = METHODS[method]
    if preset["backend"] is None:
        return oa.RandomSampler()
    if preset["backend"] == "tpe":
        raise ValueError("TPE runs through Optuna's study API, not optim-agent's sampler API")
    return oa.AgentSampler(
        backend=preset["backend"], model=model or preset["model"], effort=effort,
        context="Full MNIST CNN validation error; tune learning rate, batch size, dropout and width.",
        n_init=4, timeout=timeout, seed=seed,
    )


def download():
    train, test = _datasets(download=True)
    print(f"MNIST ready at {DATA}: train={len(train)} test={len(test)}")


def _run_tpe(seed, trials, epochs, workers, gpus):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=4))
    records = []
    before = 0
    print(f"== TPE seed {seed}: {before}/{trials} trials present, "
          f"epochs={epochs}, workers={workers}, gpus={gpus or ['cpu']} ==")
    objective = _objective(epochs, seed, gpus)

    def wrapped(trial):
        t = _OptunaTrialAdapter(trial)
        value = objective(t)
        records.append(_trial_record(t, t._mnist_metrics))
        return value

    study.optimize(wrapped, n_trials=trials, n_jobs=workers)
    records.sort(key=lambda r: r["trial"])
    return records, study.best_value, study.best_params


def run(method, seeds, trials, epochs, workers, gpus, effort, timeout, model):
    if gpus:
        _ensure_cuda_driver_compat()
        if not _cuda_ready():
            raise SystemExit("CUDA GPUs were requested but PyTorch cannot initialize CUDA")
    if method not in METHODS:
        raise SystemExit(f"unknown method {method!r}; choose one of {', '.join(METHODS)}")
    _datasets(download=False)
    ASSETS.mkdir(parents=True, exist_ok=True)
    STORAGE.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        label = _run_label(method, effort)
        safe = _sanitize_label(label)
        if method == "TPE":
            records, best_value, best_params = _run_tpe(seed, trials, epochs, workers, gpus)
        else:
            db = STORAGE / f"mnist_{safe}_s{seed}.json"
            sampler = _sampler(method, seed, effort, timeout, model)
            study = oa.create_study(direction="minimize", sampler=sampler, storage=db,
                                    seed=seed, max_concurrency=workers)
            before = len(study.trials)
            print(f"== {method} seed {seed}: {before}/{trials} trials present, "
                  f"epochs={epochs}, workers={workers}, gpus={gpus or ['cpu']} ==")
            if before < trials:
                study.optimize(_objective(epochs, seed, gpus), n_trials=trials - before)
            records = []
            for t in study.trials:
                metrics = getattr(t, "_mnist_metrics", None) or {
                    "test_error": t.value, "test_acc": None, "test_loss": None,
                    "history": [{"step": s, "test_error": v} for s, v in t.intermediate],
                }
                records.append(_trial_record(t, metrics))
            best_value, best_params = study.best_value, study.best_params
        out = {
            "label": label, "method": method, "effort": effort,
            "seed": seed, "epochs": epochs, "trials": trials,
            "workers": workers, "gpus": gpus, "records": records,
            "best_error": best_value, "best_params": best_params,
        }
        path = ASSETS / f"mnist_curves_{safe}_s{seed}.json"
        path.write_text(json.dumps(out, indent=1))
        print(f"wrote {path} best_error={best_value:.4g}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    by_label = {}
    for path in sorted(ASSETS.glob("mnist_curves_*_s*.json")):
        run_data = json.loads(path.read_text())
        by_label.setdefault(run_data["label"], []).append(run_data)
    if not by_label:
        raise SystemExit("no mnist_curves_*_s*.json in docs/assets — run an experiment first")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for label in PLOT_LABELS:
        runs = by_label.get(label)
        if not runs:
            continue
        curves = [_best_error_curve(r["records"]) for r in runs]
        width = min(len(c) for c in curves)
        mean = np.mean([c[:width] for c in curves], axis=0)
        method = next((m for m, p in METHODS.items() if label == p.get("label") or label.startswith(p.get("label", m) + "-")),
                      label.split("-")[0])
        p = METHODS.get(method, {})
        p = {**p, **PLOT_STYLES.get(label, {})}
        ax.plot(range(1, width + 1), mean, marker="o", ms=4, lw=1.8,
                color=p.get("color"), linestyle=p.get("style", "solid"),
                label=f"{label} (n={len(runs)})")
    ax.set_xlabel("trial")
    ax.set_ylabel("best test error % (mean over seeds)")
    ax.set_title("MNIST CNN hyperparameter optimization")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = ASSETS / "mnist_benchmarks.png"
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def selfcheck():
    assert _sanitize_label("agent/mock") == "agent-mock"
    assert _best_error_curve([{"test_error": 4}, {"test_error": 5}, {"test_error": 3}]) == [4, 4, 3]
    assert _device_for_trial(9, [0, 1, 2, 3]) == "cuda:1"
    torch, _, _ = _torch()
    model = SmallCNN.make(16, 0.1)
    with torch.no_grad():
        y = model(torch.zeros(2, 1, 28, 28))
    assert tuple(y.shape) == (2, 10)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "one.json"
        p.write_text(json.dumps({"ok": True}))
        assert json.loads(p.read_text())["ok"] is True
    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    p_run = sub.add_parser("run")
    p_run.add_argument("--method", required=True, choices=list(METHODS))
    p_run.add_argument("--seeds", type=int, nargs="+", default=[0])
    p_run.add_argument("--trials", type=int, default=24)
    p_run.add_argument("--epochs", type=int, default=3)
    p_run.add_argument("--workers", type=int, default=8)
    p_run.add_argument("--gpus", type=int, nargs="*", default=list(range(8)))
    p_run.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p_run.add_argument("--timeout", type=float, default=600)
    p_run.add_argument("--model")
    sub.add_parser("plot")
    sub.add_parser("selfcheck")
    args = ap.parse_args()
    if args.cmd == "download":
        download()
    elif args.cmd == "run":
        run(args.method, args.seeds, args.trials, args.epochs, args.workers,
            args.gpus, args.effort, args.timeout, args.model)
    elif args.cmd == "plot":
        plot()
    else:
        selfcheck()
