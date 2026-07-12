import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optim_agent as oa
from scripts import verify_classification_cumulative_error as classification_cumulative_error


def test_classification_worker_accepts_random_baseline(monkeypatch, tmp_path):
    calls = []
    module = SimpleNamespace(
        _sampler=lambda *args: oa.RandomSampler(),
        run=lambda *args: calls.append(args),
        ASSETS=None,
        STORAGE=None,
    )
    monkeypatch.setattr(classification_cumulative_error, "_dataset_module", lambda _: module)

    args = SimpleNamespace(
        dataset="mnist",
        method="Random",
        seed=0,
        assets=str(tmp_path / "assets"),
        storage=str(tmp_path / "storage"),
        gpus=[],
    )

    classification_cumulative_error._worker(args)

    assert len(calls) == 1
    assert module.ASSETS == tmp_path / "assets"
    assert module.STORAGE == tmp_path / "storage"
