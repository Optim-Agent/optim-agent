"""Capsys tests for per-trial table/line logging (R5-R8, R12, R16; A2, A3).

TTY-dependent rendering is exercised by monkeypatching `sys.stdout.isatty` and
`shutil.get_terminal_size`; the non-TTY fallback test runs under unpatched capsys.
No real agent CLI is involved (RandomSampler studies only).
"""

import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optim_agent as oa
from optim_agent.reporting import ELLIPSIS, TableReporter, format_line


def _fake_tty(monkeypatch, columns=120):
    """Make stdout look like a terminal of `columns` chars wide (R16)."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(shutil, "get_terminal_size",
                        lambda fallback=None: os.terminal_size((columns, 24)))


def _two_param(trial):
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", 0, 10)
    return (x - 2) ** 2 + y


def test_table_columns_in_declaration_order(monkeypatch, capsys):
    _fake_tty(monkeypatch)
    study = oa.create_study(seed=0)
    study.optimize(_two_param, n_trials=3, verbose="table")
    lines = capsys.readouterr().out.strip().splitlines()
    header, rows = lines[0], lines[1:]
    for name in ("trial", "value", "best", "state", "x", "y"):
        assert name in header
    # fixed columns first (trial, value, best, state), params in first-seen order (R6)
    assert header.index("trial") < header.index("value") < header.index("best")
    assert header.index("best") < header.index("state") < header.index("x")
    assert header.index("x") < header.index("y")
    assert len(rows) == 3  # header printed once, then exactly one row per trial
    assert rows[0].split(" | ")[3].strip() == "complete"
    assert all(line.count(" | ") == 5 for line in lines)  # six columns everywhere


def test_table_header_reprints_when_new_param_appears(monkeypatch, capsys):
    _fake_tty(monkeypatch)

    def objective(trial):  # define-by-run: a param joins mid-study (R12)
        x = trial.suggest_float("x", -5, 5)
        if trial.number >= 2:
            return x + trial.suggest_float("late", 0, 1)
        return x

    study = oa.create_study(seed=0)
    study.optimize(objective, n_trials=4, verbose="table")
    lines = capsys.readouterr().out.strip().splitlines()
    headers = [line for line in lines if line.startswith("trial")]
    assert len(headers) == 2  # reprint when the column set grows
    assert "late" not in headers[0]
    assert "late" in headers[1]
    assert headers[1].index("x") < headers[1].index("late")
    assert len(lines) == 2 + 4  # two headers + one row per trial


def test_table_truncates_to_narrow_terminal(monkeypatch, capsys):
    _fake_tty(monkeypatch, columns=30)

    def objective(trial):
        trial.suggest_categorical("some_long_parameter_name",
                                  ["an-extremely-long-categorical-choice-value"])
        return 1.0

    study = oa.create_study(seed=0)
    study.optimize(objective, n_trials=2, verbose="table")
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3  # header + two rows; no wrapping onto extra lines
    assert all(len(line) <= 30 for line in lines)
    assert any(ELLIPSIS in line for line in lines)


def test_non_tty_falls_back_to_line_format(capsys):
    # R8/R16: unpatched capsys stdout is not a TTY, so even verbose="table"
    # must emit the greppable legacy line format.
    study = oa.create_study(seed=0)
    study.optimize(_two_param, n_trials=2, verbose="table")
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert all(line.startswith("[optim-agent] trial ") for line in lines)
    assert all("value=" in line and "best=" in line for line in lines)


def test_verbose_true_maps_to_table_on_tty(monkeypatch, capsys):
    _fake_tty(monkeypatch)
    study = oa.create_study(seed=0)
    study.optimize(_two_param, n_trials=1)  # default verbose=True (D7)
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("trial")
    assert len(lines) == 2


def test_verbose_line_forces_line_format_even_on_tty(monkeypatch, capsys):
    _fake_tty(monkeypatch)
    study = oa.create_study(seed=0)
    study.optimize(_two_param, n_trials=2, verbose="line")
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert all(line.startswith("[optim-agent] trial ") for line in lines)


def test_verbose_false_is_silent(monkeypatch, capsys):
    _fake_tty(monkeypatch)
    study = oa.create_study(seed=0)
    study.optimize(_two_param, n_trials=2, verbose=False)
    assert capsys.readouterr().out == ""


def test_verbose_rejects_unknown_mode():
    study = oa.create_study(seed=0)
    with pytest.raises(ValueError, match="verbose"):
        study.optimize(lambda trial: 1.0, n_trials=1, verbose="grid")


def test_missing_and_failed_cells_render_dash(monkeypatch, capsys):
    _fake_tty(monkeypatch)

    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        if trial.number == 0:
            trial.suggest_float("extra", 0, 1)  # only trial 0 carries this param
        if trial.number == 2:
            raise RuntimeError("boom")
        return x

    study = oa.create_study(seed=0)
    study.optimize(objective, n_trials=3, catch=(RuntimeError,), verbose="table")
    lines = capsys.readouterr().out.strip().splitlines()
    rows = lines[1:]
    assert len(rows) == 3
    cells = [[cell.strip() for cell in row.split(" | ")] for row in rows]
    assert cells[0][5] != "-"                    # trial 0 has an "extra" value
    assert cells[1][5] == "-"                    # trial 1 lacks the "extra" param
    assert cells[2][1] == "-"                    # failed trial: value is None
    assert cells[2][3] == "failed"
    assert cells[2][5] == "-"


def test_format_line_matches_legacy_shape():
    done = SimpleNamespace(number=3, value=0.25, state="complete")
    assert format_line(done, 0.25) == "[optim-agent] trial 3: value=0.25 state=complete best=0.25"
    failed = SimpleNamespace(number=4, value=None, state="failed")
    assert format_line(failed, None) == "[optim-agent] trial 4: state=failed"


def test_reporter_columns_grow_from_seen_trial_params(capsys):
    reporter = TableReporter(width=80)
    assert reporter.columns == ["trial", "value", "best", "state"]
    t0 = SimpleNamespace(number=0, params={"x": 1.0}, value=0.5, state="complete")
    reporter.report(t0, 0.5)
    t1 = SimpleNamespace(number=1, params={"x": 0.5, "y": 2}, value=0.4, state="complete")
    reporter.report(t1, 0.4)
    t2 = SimpleNamespace(number=2, params={"x": 0.1}, value=0.3, state="complete")
    reporter.report(t2, 0.3)
    lines = capsys.readouterr().out.strip().splitlines()
    # columns come from the ordered union of seen trial.params keys only (R12)
    assert reporter.columns == ["trial", "value", "best", "state", "x", "y"]
    assert len(lines) == 5  # header, row, header (new key), row, row
    assert "y" not in lines[0]
    assert "y" in lines[2]
    assert lines[4].split(" | ")[-1].strip() == "-"  # t2 never saw "y"
