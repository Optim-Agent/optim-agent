# Roadmap

optim-agent aims to make coding agents useful as auditable, small-budget
optimizers for expensive black-box objectives.

## Public launch

- Make installation, offline evaluation, and the first optimization run take
  less than five minutes.
- Publish reproducible benchmark data and an optimization-trajectory demo.
- Add practical examples for classical ML, quant research, and inference
  quality/latency/cost trade-offs.
- Establish contribution, security, release, and documentation workflows.

## Research

- Compare more agent and classical optimizers under matched budgets.
- Measure token cost, wall-clock overhead, robustness, and sensitivity to
  semantic context.
- Explore constrained, conditional, and multi-objective optimization.
- Study safe early stopping and diverse parallel proposal portfolios.

## Product directions

- Exportable study tables and built-in convergence visualizations.
- Optional MLflow and Weights & Biases logging recipes.
- Callbacks and structured trial metadata.
- A stable backend adapter contract for additional agent CLIs.

## Non-goals

- Replacing mature high-budget Bayesian optimization systems in every regime.
- Hiding provider cost, benchmark variance, or failed agent calls.
- Adding heavyweight dependencies to the core package.
- Executing arbitrary model-generated code without user-controlled boundaries.

The roadmap is directional rather than a release commitment. Priorities may
change based on evidence and contributor demand.
