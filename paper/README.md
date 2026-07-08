# Paper workspace

Reserved for the arXiv paper on optim-agent.

Planned outline:

1. **Introduction** — LLM agents as hyperparameter samplers: qualitative
   reasoning (what a parameter *means*) fused with quantitative reasoning
   (what the history *shows*), versus evolutionary and Bayesian strategies.
2. **Method** — sampler levels (low → max: history window, explicit reasoning,
   persistent qualitative notes, multi-candidate ranking), agent pruner levels
   (loose → tight), and the skill mode where the agent reads the target
   codebase before proposing configurations.
3. **Experiments**
   - Classic test functions (Gramacy & Lee, Himmelblau) — see `examples/`.
   - MNIST classification: CNN learning rate, batch size, dropout, width.
   - ARIMA time-series fitting: (p, d, q) order and trend selection.
   - Baselines: random search, evolutionary, Bayesian (TPE/GP).
   - Ablations: sampler level, backend model, with/without context, pruner level.
4. **Discussion** — token cost vs. trial cost trade-off; when agent sampling
   wins (expensive objectives, few trials, semantically meaningful parameters).

Experiment scripts and LaTeX sources go here as they are produced.
