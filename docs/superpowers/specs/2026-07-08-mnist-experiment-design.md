# MNIST Experiment Design

## Goal

Add a reproducible MNIST hyperparameter-optimization example that uses full MNIST, can saturate the machine's 8 GPUs with parallel trials, and writes plots plus raw results under `docs/assets`.

## Scope

- Create `examples/mnist.py`.
- Use the full torchvision MNIST train and test splits.
- Abort on dataset download failure; do not fall back to partial or synthetic data.
- Tune CNN learning rate, batch size, dropout, and width.
- Include `Random`, `mock`, and real `AgentSampler` methods so the script is usable offline and with authenticated agent CLIs.
- Save per-method/per-seed JSON files as `docs/assets/mnist_curves_<method>_s<seed>.json`.
- Save the summary figure as `docs/assets/mnist_benchmarks.png`.

## Architecture

`examples/mnist.py` follows the existing examples' CLI pattern: `download`, `run`, `plot`, and `selfcheck` subcommands. Training is a compact PyTorch CNN objective. The run path uses optim-agent with JSON storage and `max_concurrency` workers so independent trials can run concurrently across the available GPUs while preserving per-trial metrics for the output JSON.

Each trial picks one GPU by local worker slot modulo the configured GPU list. This avoids distributed-training complexity; MNIST is small, so the useful parallelism is many independent trials rather than multi-GPU data parallelism for one trial.

## Search Space

- `lr`: float `[1e-4, 3e-2]`, log scale.
- `batch_size`: categorical `[64, 128, 256, 512]`.
- `dropout`: float `[0.0, 0.6]`.
- `width`: categorical `[16, 32, 64, 96, 128]`.

The objective minimizes test error percentage after a small fixed number of epochs. The JSON records test error, accuracy, loss, parameters, state, and per-epoch validation history.

## Dependencies

PyTorch and torchvision are required for running MNIST. Matplotlib is required only for plotting. Optuna is not required for the first implementation; a TPE baseline can be added later if needed.

## Verification

- Unit-test the pure helpers without downloading MNIST or requiring CUDA.
- Run `python examples/mnist.py selfcheck`.
- Run `python tests/test_optim_agent.py`.
- Download the full MNIST dataset before the full run.
- Run the full experiment on 8 GPUs when dependencies and data are available.
