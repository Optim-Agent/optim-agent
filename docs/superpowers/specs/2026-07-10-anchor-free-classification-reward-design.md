# Anchor-Free Classification Reward Design

## Objective

Improve only the generic prompt construction and prompt-facing enrichment in
`optim_agent` until GPT-5.5 at medium reasoning effort achieves a cumulative
incumbent-best test-error reward at most 80% of the best fresh Random or TPE
reward on both MNIST and CIFAR-10.

Each candidate uses 10 trials and seeds 0 through 4. For a seed, reward is
`sum(best_test_error_so_far[i] for i in range(10))`; reported reward is the
mean over the five seeds. Lower is better.

## Fairness Boundary

- Pass `gpt-5.5` and `medium` explicitly to every Codex-backed run.
- Generate fresh Random and TPE baselines with the same trial, seed, epoch,
  worker, GPU, dataset, model, and search-space settings as the GPT runs.
- Freeze those baseline curves for prompt iterations so only GPT stochasticity
  and the focused prompt change vary.
- Do not use fixed configurations, benchmark-derived anchors, prior benchmark
  outcomes, dataset-specific numeric priors, or hidden warm starts.
- A generic advisory candidate is allowed only when it is computed from the
  current search space and observations available in that same run.

## Measurement Harness

Create `scripts/verify_classification_reward.py`. Its parent mode prepares
fresh baselines when absent, clears only the current GPT output, runs MNIST on
GPUs 0-3 and CIFAR-10 on GPUs 4-7 concurrently, and emits one final JSON line.
Its worker mode imports the selected example module, redirects `ASSETS` and
`STORAGE` into `autoresearch-results/classification-n10-s5/`, and calls the
existing `run()` function.

The final JSON contains `max_ratio`, `mnist_ratio`, `cifar10_ratio`, all six
method rewards, and per-seed rewards. `max_ratio` is the primary metric. A run
can stop only when both ratios are at most 0.8.

## Refinement Strategy

Prompt hypotheses are serial because a full verification consumes all eight
GPUs. Within each verification, the two datasets run concurrently on disjoint
four-GPU sets.

The initial strategy families are:

1. Budget-aware natural-language history that emphasizes immediate incumbent
   improvement and remaining trials.
2. A compact quantitative table including objective, incumbent, improvement,
   and parameter changes, so the model can compare trials without parsing
   Python dictionaries embedded in prose.
3. A reproducible TPE-style advisory candidate derived only from the current
   run's observed trials and shown as fallible evidence rather than selected
   automatically.

Each autoresearch iteration changes one prompt mechanism, adds or updates one
focused offline test first, commits the trial, measures both datasets, runs the
guard after an improvement, and keeps or reverts the commit mechanically.

## Guard And Stop

The guard is:

```bash
python tests/test_optim_agent.py && \
python examples/mnist.py selfcheck && \
python examples/cifar10.py selfcheck
```

The foreground loop is unbounded. It stops on both ratios at most 0.8, user
interruption, a true runtime blocker, or the autoresearch soft-blocker rule
after three unsuccessful strategy pivots. Discarded trials use `git revert`.
