# Contributing to optim-agent

Thanks for helping make agent-driven optimization easier to inspect, reproduce,
and use. Small, focused pull requests are easiest to review.

## Development setup

Use Python 3.9 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[examples,dev]"
pytest
```

Install `.[vision]` only when changing the MNIST or CIFAR-10 training paths.
The core package intentionally has no runtime dependencies.

## Pull requests

1. Open an issue before large API or benchmark changes.
2. Add a failing test before changing behavior.
3. Keep optional integrations out of the core dependency set.
4. Run `pytest` and build a wheel before submitting.
5. Explain user-visible changes and update documentation in the same PR.

Do not commit credentials, local study databases, agent transcripts, or raw
provider responses. Report security-sensitive findings using
[SECURITY.md](SECURITY.md), not a public issue.

## Benchmarks

Benchmark changes must record the model identifier, effort, context policy,
trial budget, seeds, search-space version, and baseline configuration. Generate
tables and figures from committed result data; do not hand-edit reported
numbers. Clearly distinguish exploratory runs from publication-ready results.

New agent backends should degrade safely when their CLI is missing, times out,
or returns invalid JSON. Tests must not require paid API access.

## Documentation

Examples should be runnable, deterministic where practical, and honest about
compute, data, and provider requirements. Prefer a short working example over a
framework-specific abstraction.
