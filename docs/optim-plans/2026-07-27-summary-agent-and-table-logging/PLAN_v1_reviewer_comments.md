# PLAN v1 — Reviewer Comments

Mode: Reviewer (delegated foreground run, read-only subagent)
Round: 1

## Findings and dispositions

### F1 — `summarize=True` with default sampler has no LLM config to reuse (High)

- Affects: R2, I2, I3
- Evidence: `optim_agent/study.py:148` (`self.sampler = sampler or RandomSampler()`);
  only `AgentSampler` carries `backend/model/effort` (`samplers.py:39`).
- Impact: quickstart path `create_study(summarize=True)` without a sampler hits
  AttributeError / undefined behavior.
- Disposition: **accepted**. PLAN_v2 adds `summary_backend`/`summary_model`/
  `summary_effort` args; backend resolution order: explicit `summary_backend` →
  sampler's backend if `AgentSampler` → clear `ValueError` at `create_study` time
  naming the fix.

### F2 — No `mock` backend in `agent.py`; `call_agent("mock", ...)` raises (High)

- Affects: C5, R11, I3
- Evidence: `agent.py:8` (`BACKENDS = ("claude", "codex", "opencode")`),
  `agent.py:31-32` (`_cmd` raises `ValueError` otherwise); mock exists only as
  `AgentSampler._mock` (`samplers.py:256`), never routed through `call_agent`.
  Independently verified by reading `agent.py` directly.
- Impact: C5's "everything runs on `backend="mock"`" has no implementation path for
  the summarizer as written.
- Disposition: **accepted**. PLAN_v2 gives the summarizer its own mock path
  (`backend == "mock"` returns a canned structured-JSON reply inside
  `summarizer.run_summary`), mirroring `AgentSampler._mock` rather than widening
  `call_agent`'s BACKENDS contract.

### F3 — Table header vs define-by-run space growth; `None` values (Medium)

- Affects: R6, R7, I1
- Evidence: space registers lazily in `Trial._suggest` (`study.py:104-106`);
  failed trials have `value=None` via `tell` (`study.py:181-189`).
- Impact: header printed on first trial freezes columns before later params exist;
  unhandled `None` cell rendering.
- Disposition: **accepted**. PLAN_v2: columns derive from `study.space` at each row;
  if a trial registers a new parameter, the header row reprints (superset columns);
  missing values (including `None`, params absent from a trial) render as `-`.

### F4 — Studies with zero complete trials reach the summary hook (Medium)

- Affects: R1, R3, I3
- Evidence: `best_trial` returns `None` when nothing completed (`study.py:252-254`).
- Impact: summary prompt built on `None` best config; meaningless persisted summary.
- Disposition: **accepted**. PLAN_v2: when `best_trial is None`, skip the agent call,
  print a one-line notice, persist nothing.

### F5 — i18n count is 8, not 9 (Low)

- Affects: R10, D10, I6
- Evidence: `docs/i18n/` holds exactly README_{DE,ES,FR,JA,KO,PT,RU,ZH}.md.
  Independently verified via `ls`.
- Disposition: **accepted**. PLAN_v2 corrects the count to 8.

## Verified (no finding)

- SQLite additive field is feasible without migration via the existing `meta` k/v
  table (`study.py:35`); JSON `_load` ignores unknown keys (`study.py:282-292`), so
  R4's backward-compatibility claims hold.
- All repo-evidence line references in PLAN_v1 verified accurate.
