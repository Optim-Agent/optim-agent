# Tutorials

Start with [`quickstart.ipynb`](quickstart.ipynb). It runs a complete offline
study, inspects the trial history, resumes from storage, and shows the one-line
switch from the mock backend to an authenticated agent CLI.

Related runnable examples:

- [`../examples/quickstart.py`](../examples/quickstart.py): minimal script.
- [`../examples/sklearn_tuning.py`](../examples/sklearn_tuning.py): CPU ML.
- [`../examples/inference_tuning.py`](../examples/inference_tuning.py):
  explicit AI inference quality, latency, and cost trade-offs.

The notebook defaults to the free `mock` backend. The mock is an integration
stand-in, not a competitive optimizer. Change `backend` only after the matching
CLI is installed and authenticated.
