# PLAN v2 — Reviewer Comments

Mode: Reviewer (delegated foreground run, read-only subagent)
Round: 2

Round-1 dispositions F1–F5 verified as correctly resolved in v2 by the reviewer;
`_load` unknown-key tolerance and the `meta` k/v table re-confirmed.

## Findings and dispositions

### F6 — Version bump misses plugin manifests and their pinning test (High)

- Affects: R9, I4, A6
- Evidence: `plugins/optim-agent/.claude-plugin/plugin.json:4` and
  `.codex-plugin/plugin.json:3` both carry `"0.1.1"`;
  `tests/test_plugin_manifests.py:18` asserts `== "0.1.1"`. Independently verified
  by grep.
- Impact: release ships plugin manifests at 0.1.1 beside package 0.2.0, or bumping
  them fails CI mid-execution.
- Disposition: **accepted**. v3 adds both plugin.json files and
  `tests/test_plugin_manifests.py:18` to I4 and Allowed paths.

### F7 — R12 header reprint races concurrent space mutation (Medium)

- Affects: R12, I1, C3
- Evidence: `Trial._suggest` mutates `study.space` (plain dict, `study.py:155`)
  outside the log lock (`study.py:104-106`, `211-229`).
- Impact: `RuntimeError: dictionary changed size during iteration` mid-render with
  `max_concurrency > 1`.
- Disposition: **accepted**. v3: reporter derives columns from the ordered union of
  `trial.params` keys it has already rendered (per-trial dicts, no shared mutable
  state), instead of iterating `study.space`.

### F8 — `extract_json` last-object-wins yields partial summaries bypassing fallback (Medium)

- Affects: R3, I3, A4
- Evidence: `extract_json` (`agent.py:60-92`) returns the last decodable dict;
  agents commonly emit one fenced JSON block per section.
- Impact: a summary missing earlier sections is treated as parseable and persisted.
- Disposition: **accepted**. v3 adds R15: validate required section keys
  (best config, space insights, trajectory, next steps); missing keys → treated as
  parse failure → raw-text fallback.

### F9 — capsys tests silently hit the line fallback; width uncontrollable (Low)

- Affects: R11, I5, A2
- Evidence: pytest captured stdout reports `isatty() == False`;
  `shutil.get_terminal_size` reads the real terminal/`COLUMNS`.
- Disposition: **accepted**. v3 adds R16: table tests monkeypatch `isatty` and
  `get_terminal_size` (or `COLUMNS`); a test asserting the non-TTY fallback fires
  under plain capsys.

### F10 — Non-AgentSampler summarize path underspecified (Low)

- Affects: R2, I3
- Evidence: study context lives only on `AgentSampler.context` (`samplers.py:31,40`);
  with `summary_backend` + `RandomSampler` there is no model/effort to reuse
  (sampler default effort is `"high"`, `samplers.py:31`).
- Disposition: **accepted**. v3 amends R2: context via
  `getattr(study.sampler, "context", None)`; when the sampler isn't an
  `AgentSampler`, model defaults to `None` (backend default) and effort to
  `"high"`.
