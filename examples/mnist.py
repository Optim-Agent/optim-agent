"""MNIST ResNet hyperparameter benchmark.

Full MNIST train/test splits, small ResNet, and optim-agent samplers. Independent
trials can run across all visible GPUs; this is trial parallelism, not
distributed training.

    python examples/mnist.py download
    python examples/mnist.py run --method Random --seeds 0 1 2 --workers 8 --trials 24 --gpus 0 1 2 3 4 5 6 7
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
from optim_agent import space as oa_space

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
DATA = ROOT / "data" / "mnist"
STORAGE = ROOT / ".optim-agent-runs" / "mnist"

METHODS = {
    "Random": dict(backend=None, model=None, color="#9ca3af", style="solid"),
    "TPE": dict(backend="tpe", model=None, color="#111827", style=(0, (2, 2))),
    "mock": dict(backend="mock", model=None, color="#2563eb", style=(0, (4, 2))),
    "codex": dict(backend="codex", model=None, label="GPT-5.5", color="#10a37f", style=(0, (6, 2))),
    "codex-no-context": dict(backend="codex", model=None, label="GPT-5.5", no_context=True,
                             color="#f97316", style=(0, (1, 1.5))),
    "claude": dict(backend="claude", model=None, color="#8b5cf6", style="solid"),
    "opencode": dict(backend="opencode", model=None, color="#16a34a", style=(0, (1, 1.5))),
}
BATCHES = [64, 128, 256, 512]
STAGE1_WIDTHS = [8, 16, 24, 32, 48, 64]
STAGE2_WIDTHS = [16, 32, 48, 64, 96, 128]
STAGE3_WIDTHS = [32, 64, 96, 128, 160, 192]
DEPTHS = [1, 2, 3]
SHIFTS = [0, 1, 2, 3]
ROTATIONS = [0, 5, 10]
SEARCH_SPACE = {
    "lr": oa_space.Float(1e-5, 5e-2, log=True,
                         context="AdamW learning rate for MNIST ResNet"),
    "batch_size": oa_space.Categorical(
        BATCHES, context="mini-batch size; larger improves GPU use but may need a larger learning rate"),
    "weight_decay": oa_space.Float(1e-6, 1e-2, log=True,
                                   context="AdamW weight decay regularization"),
    "label_smoothing": oa_space.Float(0.0, 0.2,
                                      context="cross-entropy label smoothing"),
    "stage1_width": oa_space.Categorical(STAGE1_WIDTHS,
                                          context="channel count for the first residual stage"),
    "stage2_width": oa_space.Categorical(STAGE2_WIDTHS,
                                          context="channel count for the second residual stage"),
    "stage3_width": oa_space.Categorical(STAGE3_WIDTHS,
                                          context="channel count for the third residual stage"),
    "stage1_depth": oa_space.Categorical(DEPTHS,
                                          context="residual blocks in the first stage"),
    "stage2_depth": oa_space.Categorical(DEPTHS,
                                          context="residual blocks in the second stage"),
    "stage3_depth": oa_space.Categorical(DEPTHS,
                                          context="residual blocks in the third stage"),
    "stem_dropout": oa_space.Float(0.0, 0.4, context="dropout after the input stem"),
    "stage1_dropout": oa_space.Float(0.0, 0.5,
                                     context="dropout inside first-stage residual blocks"),
    "stage2_dropout": oa_space.Float(0.0, 0.6,
                                     context="dropout inside second-stage residual blocks"),
    "head_dropout": oa_space.Float(0.0, 0.8,
                                   context="dropout before the classifier head"),
    "aug_shift": oa_space.Categorical(
        SHIFTS, context="random translation radius in pixels; 0 disables shift augmentation"),
    "aug_rotate": oa_space.Categorical(
        ROTATIONS, context="random rotation range in degrees; 0 disables rotation augmentation"),
}
PLOT_LABELS = ("Random", "TPE", "GPT-5.5-medium", "GPT-5.5-medium-no-context")
PLOT_STYLES = {
    "GPT-5.5-medium": dict(color="#009E73", style=(0, (4, 2))),
    "GPT-5.5-medium-no-context": dict(color="#D55E00", style=(0, (1, 1.5))),
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
    if preset["backend"] in (None, "tpe"):
        return method
    suffix = f"-{effort}" + ("-no-context" if preset.get("no_context") else "")
    return f"{preset.get('label', method)}{suffix}"


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


class ResNet:
    """Factory wrapper so importing this module does not require torch."""

    @staticmethod
    def make(stage1_width, stage2_width, stage3_width, stage1_depth, stage2_depth,
             stage3_depth, stem_dropout, stage1_dropout, stage2_dropout, head_dropout):
        torch, nn, F = _torch()

        class Block(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.drop = nn.Dropout2d(float(dropout))
                self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)
                self.skip = nn.Identity() if stride == 1 and in_ch == out_ch else nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                )

            def forward(self, x):
                y = F.relu(self.bn1(self.conv1(x)))
                y = self.drop(y)
                y = self.bn2(self.conv2(y))
                return F.relu(y + self.skip(x))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(1, stage1_width, 3, padding=1, bias=False),
                    nn.BatchNorm2d(stage1_width),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(float(stem_dropout)),
                )
                self.layer1 = self._stage(stage1_width, stage1_width, stage1_depth, stride=1,
                                          dropout=stage1_dropout)
                self.layer2 = self._stage(stage1_width, stage2_width, stage2_depth, stride=2,
                                          dropout=stage2_dropout)
                self.layer3 = self._stage(stage2_width, stage3_width, stage3_depth, stride=2,
                                          dropout=0.0)
                self.drop = nn.Dropout(float(head_dropout))
                self.fc = nn.Linear(stage3_width, 10)

            @staticmethod
            def _stage(in_ch, out_ch, depth, stride, dropout):
                return nn.Sequential(
                    Block(in_ch, out_ch, stride=stride, dropout=dropout),
                    *[Block(out_ch, out_ch, dropout=dropout) for _ in range(depth - 1)],
                )

            def forward(self, x):
                x = self.layer3(self.layer2(self.layer1(self.stem(x))))
                x = F.adaptive_avg_pool2d(x, 1).flatten(1)
                x = self.drop(x)
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
    x = torch.as_tensor(dataset.data).unsqueeze(1).float().div_(255.0)
    y = torch.as_tensor(dataset.targets).long()
    return torch.utils.data.TensorDataset(x, y)


def _augment(x, shift, rotate):
    torch, _, F = _torch()
    theta = torch.eye(2, 3, device=x.device, dtype=x.dtype).repeat(x.size(0), 1, 1)
    if rotate:
        angles = (torch.rand(x.size(0), device=x.device, dtype=x.dtype) * 2 - 1) * (math.pi * rotate / 180.0)
        c, s = torch.cos(angles), torch.sin(angles)
        theta[:, 0, 0], theta[:, 0, 1] = c, -s
        theta[:, 1, 0], theta[:, 1, 1] = s, c
    if shift:
        offset = (torch.rand(x.size(0), 2, device=x.device, dtype=x.dtype) * 2 - 1) * (2 * shift / 28.0)
        theta[:, :, 2] = offset
    if shift or rotate:
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    return x


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
    model = ResNet.make(
        int(params["stage1_width"]), int(params["stage2_width"]), int(params["stage3_width"]),
        int(params["stage1_depth"]), int(params["stage2_depth"]), int(params["stage3_depth"]),
        float(params["stem_dropout"]), float(params["stage1_dropout"]),
        float(params["stage2_dropout"]), float(params["head_dropout"]),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"])
    )
    aug_shift, aug_rotate = int(params["aug_shift"]), int(params["aug_rotate"])
    label_smoothing = float(params["label_smoothing"])
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = _augment(x, aug_shift, aug_rotate)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y, label_smoothing=label_smoothing).backward()
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


def _objective(epochs, seed, gpus, use_context=True):
    lock = threading.Lock()
    next_slot = 0
    ctx = (lambda text: text if use_context else None)

    def objective(trial):
        nonlocal next_slot
        with lock:
            slot = next_slot
            next_slot += 1
        lr = trial.suggest_float("lr", 1e-5, 5e-2, log=True,
                                 context=ctx("AdamW learning rate for MNIST ResNet"))
        batch_size = trial.suggest_categorical(
            "batch_size", BATCHES,
            context=ctx("mini-batch size; larger improves GPU use but may need a larger learning rate"),
        )
        weight_decay = trial.suggest_float(
            "weight_decay", 1e-6, 1e-2, log=True,
            context=ctx("AdamW weight decay regularization"),
        )
        label_smoothing = trial.suggest_float(
            "label_smoothing", 0.0, 0.2,
            context=ctx("cross-entropy label smoothing"),
        )
        stage1_width = trial.suggest_categorical(
            "stage1_width", STAGE1_WIDTHS,
            context=ctx("channel count for the first residual stage"),
        )
        stage2_width = trial.suggest_categorical(
            "stage2_width", STAGE2_WIDTHS,
            context=ctx("channel count for the second residual stage"),
        )
        stage3_width = trial.suggest_categorical(
            "stage3_width", STAGE3_WIDTHS,
            context=ctx("channel count for the third residual stage"),
        )
        stage1_depth = trial.suggest_categorical(
            "stage1_depth", DEPTHS,
            context=ctx("residual blocks in the first stage"),
        )
        stage2_depth = trial.suggest_categorical(
            "stage2_depth", DEPTHS,
            context=ctx("residual blocks in the second stage"),
        )
        stage3_depth = trial.suggest_categorical(
            "stage3_depth", DEPTHS,
            context=ctx("residual blocks in the third stage"),
        )
        stem_dropout = trial.suggest_float(
            "stem_dropout", 0.0, 0.4,
            context=ctx("dropout after the input stem"),
        )
        stage1_dropout = trial.suggest_float(
            "stage1_dropout", 0.0, 0.5,
            context=ctx("dropout inside first-stage residual blocks"),
        )
        stage2_dropout = trial.suggest_float(
            "stage2_dropout", 0.0, 0.6,
            context=ctx("dropout inside second-stage residual blocks"),
        )
        head_dropout = trial.suggest_float(
            "head_dropout", 0.0, 0.8,
            context=ctx("dropout before the classifier head"),
        )
        aug_shift = trial.suggest_categorical(
            "aug_shift", SHIFTS,
            context=ctx("random translation radius in pixels; 0 disables shift augmentation"),
        )
        aug_rotate = trial.suggest_categorical(
            "aug_rotate", ROTATIONS,
            context=ctx("random rotation range in degrees; 0 disables rotation augmentation"),
        )
        device = _device_for_trial(slot, gpus)
        metrics = _train_once(dict(
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, label_smoothing=label_smoothing,
            stage1_width=stage1_width, stage2_width=stage2_width, stage3_width=stage3_width,
            stage1_depth=stage1_depth, stage2_depth=stage2_depth, stage3_depth=stage3_depth,
            stem_dropout=stem_dropout, stage1_dropout=stage1_dropout,
            stage2_dropout=stage2_dropout, head_dropout=head_dropout,
            aug_shift=aug_shift, aug_rotate=aug_rotate,
        ), device, epochs, seed + slot)
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
        context=(None if preset.get("no_context") else
                 "Full MNIST neural architecture search with cumulative best-so-far error: "
                 "minimize the sum of "
                 "incumbent best test errors over the trial budget. Tune learning rate, batch size, weight "
                 "decay, label smoothing, stage widths, stage depths, stage dropouts, translation "
                 "and rotation augmentation."),
        n_init=4, timeout=timeout, seed=seed,
        initial_space=(None if preset.get("no_context") else SEARCH_SPACE),
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
                study.optimize(_objective(epochs, seed, gpus, not METHODS[method].get("no_context")),
                               n_trials=trials - before)
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
            "model": model if METHODS[method]["backend"] == "codex" else None,
            "seed": seed, "epochs": epochs, "trials": trials,
            "workers": workers, "gpus": gpus, "records": records,
            "best_error": best_value, "best_params": best_params,
        }
        path = ASSETS / f"mnist_curves_{safe}_s{seed}.json"
        path.write_text(json.dumps(out, indent=1))
        print(f"wrote {path} best_error={best_value:.4g}")


def _load_plot_runs():
    by_label = {}
    for path in sorted(ASSETS.glob("mnist_curves_*_s*.json")):
        run_data = json.loads(path.read_text())
        by_label.setdefault(run_data["label"], []).append(run_data)
    if set(by_label) != set(PLOT_LABELS):
        raise SystemExit(f"MNIST plot requires exactly {PLOT_LABELS}")
    methods = dict(zip(PLOT_LABELS, ("Random", "TPE", "codex", "codex-no-context")))
    for label, runs in by_label.items():
        if len(runs) != 5 or {run.get("seed") for run in runs} != set(range(5)):
            raise SystemExit(f"MNIST plot requires seeds 0..4 for {label}")
        for run in runs:
            if (run.get("method") != methods[label] or run.get("trials") != 10
                    or len(run.get("records", ())) != 10
                    or any(record.get("state", "complete") != "complete"
                           or record.get("test_error") is None for record in run["records"])
                    or (label.startswith("GPT")
                        and (run.get("model") != "gpt-5.5" or run.get("effort") != "medium"))):
                raise SystemExit(f"incompatible MNIST plot data for {label}")
    return by_label


def plot():
    from scripts.render_classification_benchmarks import render

    render()


def selfcheck():
    assert _sanitize_label("agent/mock") == "agent-mock"
    assert _best_error_curve([{"test_error": 4}, {"test_error": 5}, {"test_error": 3}]) == [4, 4, 3]
    assert _device_for_trial(9, [0, 1, 2, 3]) == "cuda:1"
    torch, _, _ = _torch()
    model = ResNet.make(16, 32, 64, 1, 1, 1, 0.1, 0.1, 0.1, 0.1)
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
    p_run.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p_run.add_argument("--trials", type=int, default=24)
    p_run.add_argument("--epochs", type=int, default=3)
    p_run.add_argument("--workers", type=int, default=8)
    p_run.add_argument("--gpus", type=int, nargs="*", default=list(range(8)))
    p_run.add_argument("--effort", default="high", choices=["low", "medium", "high"])
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
