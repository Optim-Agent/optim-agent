# PLAN v2 — Summary Agent and Table Logging

Run: `40b13e0d7b8846908e7b20a8b5466f5f`
Repo: `/Users/zhuofan/ClaudeExp/optim-agent`
Date: 2026-07-27
Depth: big-plan (10 planning questions answered)
Supersedes: PLAN_v1.md (revision ledger at end; reviewer comments in
`PLAN_v1_reviewer_comments.md`)

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
- **R2**: Summary backend resolution order: explicit `summary_backend` arg → the
  sampler's `backend` if the sampler is an `AgentSampler` → raise a clear
  `ValueError` at `create_study` time naming the fix (pass `summary_backend=...`).
  `summary_model`/`summary_effort` override the sampler's model/effort when given;
  otherwise the sampler's values are reused. Resolution happens at `create_study`,
  not at summary time, so misconfiguration fails fast.
- **R3**: Summary is structured into fixed sections: best configuration + value,
  search space insights, trajectory highlights, suggested next steps. Agent replies
  are parsed with the existing `extract_json` (`optim_agent/agent.py:60`);
  unparseable replies fall back to printing the raw text (never raise, never fail
  the study).
- **R4**: Summary is printed to the terminal AND persisted on the study
  (`study.summary`), stored in JSON storage (additive key; `_load` already ignores
  unknown keys, `study.py:282-292`) and in SQLite via the existing `meta` k/v table
  (`study.py:35`) — no schema migration. Old storages load with `summary=None`.
- **R5**: `Study.optimize(verbose=...)` accepts `"table"` | `"line"` | `False`;
  `True` maps to `"table"` (backward compatible). Default stays `verbose=True`.
- **R6**: Table columns: trial #, value, best-so-far, state, plus one column per
  declared search-space parameter, in parameter declaration order.
- **R7**: Pure-stdlib renderer: `shutil.get_terminal_size` for width, truncation of
  long values (ellipsis) to fit; no wrapping, no in-place redraw.
- **R8**: Non-TTY stdout (piped/CI) automatically falls back to the existing line
  format so logs stay greppable.
- **R9**: Version `0.1.1 → 0.2.0` in `pyproject.toml:7`, `optim_agent/__init__.py:7`,
  `CHANGELOG.md` (new `## 0.2.0` section), `docs/index.html:75`.
- **R10**: Documentation in English README **and all 8 i18n READMEs**
  (`docs/i18n/README_{DE,ES,FR,JA,KO,PT,RU,ZH}.md`), covering both features.
- **R11**: Tests: capsys-based tests for table rendering (columns, truncation,
  non-TTY fallback, `verbose="line"` compat) and summary-agent tests via `mock`
  (structured parse, fallback on garbage, persistence round-trip, off by default).
- **R12**: Define-by-run handling: table columns derive from `study.space` at each
  row; if a trial registers a new parameter, the header row reprints with the
  superset of columns. Missing values (`value=None` for failed trials, params absent
  from a given trial) render as `-`.
- **R13**: Summarizer has its own mock path: `run_summary(..., backend="mock")`
  returns a canned structured-JSON reply inside `summarizer.py`, mirroring
  `AgentSampler._mock` (`samplers.py:256`). `agent.py`'s `BACKENDS`/`call_agent`
  contract is unchanged.
- **R14**: When `best_trial is None` (zero complete trials — e.g. `n_trials=0` or all
  failed/pruned), the summary hook skips the agent call, prints a one-line notice,
  and persists nothing.

## Constraints

- **C1**: Python ≥ 3.9 (pyproject `requires-python`); stdlib only.
- **C2**: Ruff config: `line-length = 100`, lint `E4,E7,E9,F`, target `py39`.
- **C3**: Thread-safety: `optimize` runs trials under a lock for logging
  (`study.py:227-229`); table row printing must respect the same lock.
- **C4**: Storage formats: JSON (`Study._save/_load`, `study.py:270-292`) and SQLite
  (`_SQLiteStore`, `study.py:19-90`). Summary persistence must be additive and
  backward compatible with existing stored studies.
- **C5**: No test may require a real agent CLI; everything runs on `mock` paths
  (sampler `_mock`, summarizer mock per R13).

## Repo evidence

- Per-trial logging is a single `print` in `Study._log`, `optim_agent/study.py:242-246`,
  called under a lock at `study.py:227-229`, gated by `verbose=True`. No `logging`
  module usage anywhere in the package.
- Agent invocation: `call_agent(backend, model, prompt, timeout, effort, cwd)`,
  `optim_agent/agent.py:40`; JSON extraction `extract_json`, `agent.py:60`.
  `BACKENDS = ("claude", "codex", "opencode")` (`agent.py:8`); `mock` exists only as
  `AgentSampler._mock` (`samplers.py:256`).
- Sampler prompt pattern to imitate: `AgentSampler._prompt`, `samplers.py:130-214`;
  validation/fallback pattern: `AgentSampler._validate_reply`, `samplers.py:218`.
- Results: `Study.trials` list (`study.py:154`); `Trial` fields `number`, `params`,
  `value`, `state`, `intermediate` (`study.py:93-101`); best properties
  `best_trial`/`best_value`/`best_params` (`study.py:250-266`); `best_trial` is
  `None` with zero complete trials (`study.py:252-254`).
- Search space registers lazily during trials: `Trial._suggest`, `study.py:104-106`.
- `Study.__init__` defaults `sampler = sampler or RandomSampler()` (`study.py:148`).
- `optimize` returns `self` (`study.py:240`) — summary hook lands at the end of
  `optimize`.
- Version strings: `pyproject.toml:7`, `optim_agent/__init__.py:7`, `CHANGELOG.md:9`,
  `docs/index.html:75`.
- Tests: `tests/test_optim_agent.py` (1762 lines); no capsys coverage of `_log`
  today; `test_mock_backend_and_storage` (line 1171) is the persistence test pattern.
- i18n READMEs: exactly 8 files in `docs/i18n/` (DE, ES, FR, JA, KO, PT, RU, ZH).
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
- **D10** (docs/tests): English + all 8 i18n READMEs; capsys + mock tests. [user]

## Implementation items

- **I1**: `optim_agent/reporting.py` (new): stdlib table renderer —
  `TableReporter` with header derived from current `study.space` (reprinted on space
  growth per R12), one row per trial, width budgeting via
  `shutil.get_terminal_size`, ellipsis truncation; `-` for missing/`None` cells;
  `format_line(trial, best)` for the existing line format (extracted from
  `Study._log`).
- **I2**: `optim_agent/study.py`: `verbose` accepts `"table"|"line"|False` (`True`→
  `"table"`); TTY detection decides table vs line fallback (R8); `create_study(...,
  summarize=False, summary_backend=None, summary_model=None, summary_effort=None)`
  with backend resolution per R2 at construction; end-of-`optimize` summary hook
  honoring R14; `study.summary` attribute; `_save`/`_load` additive key and
  `_SQLiteStore` `meta` k/v entry (R4).
- **I3**: `optim_agent/summarizer.py` (new): `build_summary_prompt(study)` and
  `run_summary(study, backend, model, effort)` following the
  `AgentSampler._prompt`/`call_agent`/`extract_json` patterns; structured sections
  (best config, space insights, trajectory, next steps); raw-text fallback (R3);
  `backend="mock"` canned reply (R13).
- **I4**: Version bump `0.1.1 → 0.2.0`: `pyproject.toml`, `optim_agent/__init__.py`,
  `CHANGELOG.md`, `docs/index.html`.
- **I5**: Tests (new `tests/test_reporting.py` + additions to
  `tests/test_optim_agent.py`): table columns/order, header reprint on late param,
  truncation at narrow width, non-TTY fallback, `verbose="line"`, `verbose=False`
  silence, `-` for None/failed cells, summary parse/fallback/persistence (JSON +
  SQLite), summarize off by default, `ValueError` when backend unresolvable (R2),
  zero-complete-trials skip (R14).
- **I6**: Docs: English README (Quickstart + logging section) and 8 i18n READMEs.

## Acceptance criteria

- **A1**: `pytest` passes (full suite, including new tests) with no real CLI calls.
- **A2**: `verbose="table"` on a TTY renders header + one row per trial with all
  param columns; narrow widths truncate with ellipsis; piped output is line format;
  late-registered params trigger a header reprint.
- **A3**: `verbose=True`/`"line"`/`False` behave per D7; existing callers unaffected.
- **A4**: `create_study(summarize=True)` with mock backend prints and persists a
  structured summary; garbage agent output prints raw text without raising;
  zero-complete-trials studies skip gracefully; missing backend raises a clear
  `ValueError` at `create_study`.
- **A5**: A stored study (JSON and SQLite) round-trips `summary`; old storages load
  with `summary=None`.
- **A6**: `optim_agent.__version__ == "0.2.0"`; pyproject version `0.2.0`.
- **A7**: Ruff clean under repo config.

## Verification class

- **V1**: Unit tests (new + existing) — primary gate, mock paths only.
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

## Revision ledger (v1 → v2)

- R2 rewritten per F1: explicit backend resolution order with fail-fast `ValueError`
  for non-AgentSampler studies; `summary_backend` added to the arg set.
- R4 amended: names the SQLite `meta` k/v table (`study.py:35`) as the persistence
  mechanism; notes JSON `_load` unknown-key tolerance.
- R10/D10/I6 corrected per F5: 8 i18n READMEs, not 9.
- R12 added per F3: header reprint on define-by-run space growth; `-` for
  missing/`None` cells.
- R13 added per F2: summarizer-local mock path; `agent.py` BACKENDS unchanged.
- R14 added per F4: skip summary when `best_trial is None`.
- Repo evidence extended with `study.py:148`, `study.py:252-254`, `study.py:104-106`,
  `agent.py:8`, `samplers.py:256`.
- I1–I3, I5 updated to reference R12–R14.
- F-dispositions recorded in `PLAN_v1_reviewer_comments.md`.
