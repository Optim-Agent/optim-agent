# Changelog

All notable changes to optim-agent are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/).

## Unreleased

## 0.2.0 - 2026-07-27

### Added

- Optional summary agent: `create_study(..., summarize=True)` runs an agent once
  after `optimize()` finishes, printing and persisting a structured summary with
  the best configuration, search-space insights, trajectory highlights, and
  suggested next steps.
- Table logging: `Study.optimize(verbose="table")` renders a terminal-friendly
  table with one column per search-space parameter, width-aware truncation, and
  automatic fallback to the classic line format on non-TTY output.

## 0.1.1 - 2026-07-13

### Added

- Public project governance, security, and contribution guidance.
- Explicit development and optional vision dependency groups.

### Changed

- CI now separates lightweight core coverage from PyTorch vision coverage.
- Benchmark documentation now focuses on hard-function, classification, and
  credit-card experiments with public-facing labels.
- Agent sampler prompt controls now expose trial history length, explicit
  reasoning, and qualitative notes as arguments.

### Removed

- Removed the contextual quant benchmark, runner, figure, and JSON artifacts.

## 0.1.0 - 2026-07-08

### Added

- Optuna-style `Study` and `Trial` APIs with ask/tell and optimize workflows.
- Agent samplers for Claude Code, Codex, and OpenCode.
- Agent-guided pruning, JSON and SQLite storage, and distributed workers.
- Reproducible optimization and image-classification benchmarks.

[Unreleased]: https://github.com/Optim-Agent/optim-agent/commits/main
