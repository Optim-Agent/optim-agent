# Tutorials

- [`quickstart.ipynb`](quickstart.ipynb): run a complete offline study, then
  switch the same code to an authenticated agent CLI.
- [`../examples/sklearn_tuning.py`](../examples/sklearn_tuning.py): CPU ML.
- [`../examples/quant_walk_forward.py`](../examples/quant_walk_forward.py):
  leakage-aware walk-forward strategy tuning.
- [`../examples/inference_tuning.py`](../examples/inference_tuning.py): explicit
  AI inference quality, latency, and cost trade-offs.

Every tutorial defaults to the free `mock` backend. The mock is an integration
stand-in, not a competitive optimizer. Change `backend` only after the matching
CLI is installed and authenticated.
