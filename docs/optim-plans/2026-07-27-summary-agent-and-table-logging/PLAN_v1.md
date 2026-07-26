# PLAN v1 — Summary Agent and Table Logging

Run: `40b13e0d7b8846908e7b20a8b5466f5f`
Repo: `/Users/zhuofan/ClaudeExp/optim-agent`
Date: 2026-07-27
Depth: big-plan (10 planning questions answered)

## Goals

- **G1**: Add an optional summary agent that describes the best solution after HPO
  finishes — a differentiating feature no numerical HPO optimizer (Optuna, Hyperopt,
  Ray Tune) provides.
- **G2**: Replace/augment real-time per-trial logging with a terminal-friendly,
  line-separated, well-structured table including per-parameter columns.
- **G3**: Increment the package version to reflect the two new features.

## Non-goals

- **NG1**: No new runtime dependencies (no `rich`, no `tabulate`). Core stays
  dependency-free.
- **NG2**: No changes to the agent proposal loop (`AgentSampler._prompt`), pruners,
  storage schema beyond an additive summary field, or benchmark harnesses.
- **NG3**: No live-updating/in-place table (no ANSI cursor control); each trial appends
  one row. This keeps piped logs and CI output intact.
- **NG4**: No interactive UI, progress bars, or dashboards.

## Requirements

- **R1**: `create_study(..., summarize=False)` (opt-in). When truthy, `optimize()`
  runs the summary agent once after the last trial completes, before returning.
- **R2**: Summary agent reuses the sampler's backend/model/effort by default, with
  optional overrides (`summary_model`, `summary_effort` — exact names finalized at
  implementation; see D2).
- **R3**: Summary is structured into fixed sections: best configuration + value, search
  space insights, trajectory highlights, suggested next steps. Agent replies are
  parsed with the existing `extract_json` (`optim_agent/agent.py:60`); unparseable
  replies fall back to printing the raw text (never raise, never fail the study).
- **R4**: Summary is printed to the terminal AND persisted on the study
  (`study.summary`), stored in JSON storage and the SQLite store as an additive field.
  Old storages without the field load fine (default `None`).
- **R5**: `Study.optimize(verbose=...)` accepts `"table"` | `"line"` | `False`;
  `True` maps to `"table"` (backward compatible). Default stays `verbose=True`.
- **R6**: Table columns: trial #, value, best-so-far, state, plus one column per
  declared search-space parameter. Column order follows parameter declaration order.
- **R7**: Pure-stdlib renderer: `shutil.get_terminal_size` for width, truncation of
  long values (ellipsis) to fit; no wrapping, no in-place redraw.
- **R8**: Non-TTY stdout (piped/CI) automatically falls back to the existing line
  format so logs stay greppable.
- **R9**: Version `0.1.1 → 0.2.0` in `pyproject.toml:7`, `optim_agent/__init__.py:7`,
  `CHANGELOG.md` (new `## 0.2.0` section), `docs/index.html:75`.
- **R10**: Documentation in English README **and all 9 i18n READMEs**
  (`docs/i18n/README_{ZH,JA,KO,FR,DE,ES,PT,RU}.md` — list verified at implementation),
  covering both features.
- **R11**: Tests: capsys-based tests for table rendering (columns, truncation,
  non-TTY fallback, `verbose="line"` compat) and summary-agent tests via the `mock`
  backend (structured parse, fallback on garbage, persistence round-trip, off by
  default).

## Constraints

- **C1**: Python ≥ 3.9 (pyproject `requires-python`); no walrus-free assumptions broken,
  stdlib only.
- **C2**: Ruff config: `line-length = 100`, lint `E4,E7,E9,F`, target `py39`.
- **C3**: Thread-safety: `optimize` runs trials under a lock for logging
  (`study.py:227-229`); table row printing must respect the same lock.
- **C4**: Storage formats: JSON (`Study._save/_load`, `study.py:270-292`) and SQLite
  (`_SQLiteStore`, `study.py:19-90`). Summary persistence must be additive and
  backward compatible with existing stored studies.
- **C5**: No test may require a real agent CLI; everything runs on `backend="mock"`.

## Repo evidence

- Per-trial logging is a single `print` in `Study._log`, `optim_agent/study.py:242-246`,
  called under a lock at `study.py:227-229`, gated by `verbose=True`. No `logging`
  module usage anywhere in the package.
- Agent invocation: `call_agent(backend, model, prompt, timeout, effort, cwd)`,
  `optim_agent/agent.py:40`; JSON extraction `extract_json`, `agent.py:60`.
  Backends: `claude`, `codex`, `opencode`, plus `mock` for tests.
- Sampler prompt pattern to imitate: `AgentSampler._prompt`, `samplers.py:130-214`;
  validation/fallback pattern: `AgentSampler._validate_reply`, `samplers.py:218`.
- Results: `Study.trials` list (`study.py:154`); `Trial` fields `number`, `params`,
  `value`, `state`, `intermediate` (`study.py:93-101`); best properties
  `best_trial`/`best_value`/`best_params` (`study.py:250-266`).
- `optimize` returns `self` (`study.py:240`) — summary hook lands at the end of
  `optimize`.
- Version strings: `pyproject.toml:7`, `optim_agent/__init__.py:7`, `CHANGELOG.md:9`,
  `docs/index.html:75`.
- Tests: `tests/test_optim_agent.py` (1762 lines); no capsys coverage of `_log`
  today; `test_mock_backend_and_storage` (line 1171) is the persistence test pattern.
- No existing summary/report feature anywhere in the package.

## Web evidence (brainstorming)

- Optuna exposes only log-level control (`optuna.logging.set_verbosity`), no result
  narration: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.logging.set_verbosity.html
- Ray Tune / Optuna / Hyperopt comparisons show per-trial console reporters (Ray
  Tune CLIReporter table) as the standard for tabular trial output; none offers an
  LLM-authored result summary:
  https://druce.ai/2020/10/hyperparameter-tuning-with-xgboost-ray-tune-hyperopt-and-optuna
- HPO interpretability research treats explanation as an open add-on (e.g.
  https://arxiv.org/abs/2206.05447), supporting G1 as a differentiator.

## Resolved decisions

- **D1** (API enablement): `create_study` arg, auto-run at end of `optimize()`. [user]
- **D2** (summarizer LLM): reuse sampler backend/model/effort; optional overrides. [user]
- **D3** (output): print + persist on study (JSON + SQLite, additive). [user]
- **D4** (shape): structured sections via JSON; raw-text fallback on parse failure. [user]
- **D5** (columns): trial/value/best/state + one column per parameter. [user]
- **D6** (renderer): pure stdlib, truncate-to-width, no redraw. [user]
- **D7** (verbose API): `verbose="table"|"line"|False`, `True`→`"table"`. [user]
- **D8** (overflow/TTY): truncate to fit; non-TTY falls back to line format. [user]
- **D9** (version): `0.2.0` minor bump across pyproject, `__init__.py`, CHANGELOG,
  docs/index.html. [user]
- **D10** (docs/tests): English + all 9 i18n READMEs; capsys + mock-backend tests. [user]

## Implementation items

- **I1**: `optim_agent/reporting.py` (new): stdlib table renderer —
  `TableReporter(space, stream, width_fn)` with header row on first trial, one row
  per trial, width budgeting and ellipsis truncation; `format_line(trial, best)` for
  the existing line format (extracted from `Study._log`).
- **I2**: `optim_agent/study.py`: `verbose` accepts `"table"|"line"|False` (`True`→
  `"table"`); TTY detection decides table vs line fallback; `create_study(...,
  summarize=False, summary_model=None, summary_effort=None)`; end-of-`optimize`
  summary hook; `study.summary` attribute; `_save`/`_load` and `_SQLiteStore` additive
  summary field.
- **I3**: `optim_agent/summarizer.py` (new): `build_summary_prompt(study, context)`
  and `run_summary(study, backend, model, effort)` following the
  `AgentSampler._prompt`/`call_agent`/`extract_json` patterns; structured sections
  (best config, space insights, trajectory, next steps); raw-text fallback.
- **I4**: Version bump `0.1.1 → 0.2.0`: `pyproject.toml`, `optim_agent/__init__.py`,
  `CHANGELOG.md`, `docs/index.html`.
- **I5**: Tests in `tests/test_optim_agent.py` (or new `tests/test_reporting.py`):
  table columns/order, truncation at narrow width, non-TTY fallback, `verbose="line"`,
  `verbose=False` silence, summary parse/fallback/persistence, summarize off by
  default.
- **I6**: Docs: English README (Quickstart + logging section) and 9 i18n READMEs.

## Acceptance criteria

- **A1**: `pytest` passes (full suite, including new tests) with no real CLI calls.
- **A2**: `verbose="table"` on a TTY renders header + one row per trial with all
  param columns; narrow widths truncate with ellipsis; piped output is line format.
- **A3**: `verbose=True`/`"line"`/`False` behave per D7; existing callers unaffected.
- **A4**: `create_study(summarize=True)` with mock backend prints and persists a
  structured summary; garbage agent output prints raw text without raising.
- **A5**: A stored study (JSON and SQLite) round-trips `summary`; old storages load
  with `summary=None`.
- **A6**: `importlib.metadata.version("optim-agent")`-visible version is `0.2.0`;
  `optim_agent.__version__ == "0.2.0"`.
- **A7**: Ruff clean under repo config.

## Verification class

- **V1**: Unit tests (new + existing) — primary gate, `backend="mock"`.
- **V2**: Manual smoke: run a small random-sampler study in a real terminal and piped
  to a file to eyeball table vs line behavior.
- **V3**: Lint: `ruff check` per repo config.

## Expected evidence

- pytest output for the full suite; capsys assertions for table/line rendering;
  storage round-trip assertions; ruff output; diff of version strings.

## Dependencies

- None external. Internal: I1 before I2 (study wiring uses reporter); I3 before I2's
  summary hook; I5 after I1–I3; I4 and I6 independent.

## Allowed paths

- `optim_agent/` (study.py, new reporting.py, new summarizer.py, `__init__.py`)
- `tests/` (test_optim_agent.py and/or new test files)
- `pyproject.toml`, `CHANGELOG.md`, `docs/index.html`
- `README.md`, `docs/i18n/README_*.md`
- `docs/optim-plans/2026-07-27-summary-agent-and-table-logging/` (artifacts)
