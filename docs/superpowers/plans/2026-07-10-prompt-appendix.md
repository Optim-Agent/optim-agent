# Prompt Appendix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the paper with the current three-level prompting implementation and disclose the complete sampler and pruner prompt templates in a post-reference appendix.

**Architecture:** Keep the implementation source as the authority and change only `paper/src/main.tex`. The main method will describe current behavior, legacy experiment labels will remain explicit, and one parametric template per runtime prompt will capture conditional text without repeating near-identical rendered examples.

**Tech Stack:** AAAI 2027 LaTeX, standard `verbatim`, Bash paper verifier.

## Global Constraints

- Current sampler efforts are exactly `low`, `medium`, and `high` with history windows 5, 10, and 20.
- Reasoning/notes flags are `(no, no)`, `(yes, yes)`, and `(yes, yes)` respectively.
- Existing `xhigh` measurements retain their recorded values and are labeled legacy.
- Both runtime prompts and the sampler retry suffix must appear in the appendix.
- Add no LaTeX package solely for prompt listings.
- Preserve unrelated working-tree changes.

---

### Task 1: Align the Method and Experimental Language

**Files:**
- Modify: `paper/src/main.tex:108-116`
- Modify: `paper/src/main.tex:191-258`
- Modify: `paper/src/main.tex:298-328`
- Modify: `paper/src/main.tex:379-380`
- Modify: `paper/src/main.tex:537-624`

**Interfaces:**
- Consumes: `optim_agent.samplers.EFFORTS`, `AgentSampler._prompt`, `AgentSampler._validate_reply`, and `agent.extract_json`.
- Produces: a method description and effort table consistent with the current package, plus explicit legacy labels for old measurements.

- [ ] **Step 1: Record the stale-description baseline**

Run:

```bash
rg -n 'ranked candidates|completed, failed, and pruned|medium & 15|xhigh & all|max & all|high and above|\\emph\{max\}|serialization of intermediate|GPT-5\.5 xhigh|from low to xhigh' paper/src/main.tex
```

Expected: matches in the contribution list, sampler description, effort table, complexity statement, and experiment discussion.

- [ ] **Step 2: Correct the sampler contract and validation prose**

Replace the ranked-candidate contribution with:

```latex
\item an effort abstraction that exposes bounded history, explicit reasoning,
and persistent notes without requiring users to write prompts;
```

Replace the `AgentSampler` and structured-contract paragraphs with:

```latex
\textbf{AgentSampler.} At each non-warmup trial, the sampler serializes the
search space, parameter meanings, objective direction, and a summary drawn from
a bounded window of completed and pruned trials that have objective values. The
summary reports the incumbent, up to five promising trials, five recent trials,
and three weak trials; failed trials are not included. The agent is instructed
to return one JSON object containing every candidate value. The package extracts
the final decodable object, validates each value against the registered
distribution, clamps numeric outliers, and rejects an invalid categorical or a
missing parameter. After one corrective retry, an invalid proposal falls back
to random sampling. This makes the LLM advisory rather than trusted: the agent
proposes, the package validates, and the objective decides.

The sampler has three safeguards that are important in unattended experiments.
First, each study begins with random warmup trials, which provide exploration
and prevent the first agent call from being based on an empty table. Second,
the response parser extracts the final decodable JSON object even if a backend
wraps it in prose or a code fence. Third, invalid output never terminates the
study: the sampler retries once and then returns a random valid configuration.
These choices reduce apparent intelligence but make the optimizer robust enough
to run inside ordinary training scripts.

\textbf{Structured prompt contract.} For a distribution $\mathcal{X}_j$, the
serialized schema contains the parameter name, type, bounds or choices, log-scale
flag when relevant, and context sentence $c_j$. History entries contain trial
number, parameters, and objective value; the sampler prompt does not serialize
intermediate curves or failed-trial diagnostics. The agent is asked for
\begin{equation}
  R_t = \{(j,\tilde{x}_{t+1,j})\}_{j=1}^{d},
\end{equation}
a JSON object with one proposed value per active parameter and, when enabled,
optional \texttt{\_reasoning} and \texttt{\_note} strings. The former is ignored
after parsing; the latter is retained as a bounded qualitative scratchpad. Thus
the external agent can be stochastic and verbose while the package exposes a
validated sampler interface to the training code. Appendix~\ref{app:prompts}
reproduces the exact parametric sampler and pruner templates. The package passes
each template to the backend CLI as one flat prompt string rather than creating
separate system and user messages.
```

Also correct categorical validation to say that an invalid categorical rejects the proposal and ultimately falls back to a random valid configuration, rather than sampling only a replacement category.

- [ ] **Step 3: Replace the effort table and formal prompt context**

Use this current table:

```latex
\begin{table}[t]
\centering
\begin{tabular}{lccc}
\toprule
Effort & History window & Reason & Notes \\
\midrule
low & 5 & no & no \\
medium & 10 & yes & yes \\
high & 20 & yes & yes \\
\bottomrule
\end{tabular}
\caption{Current AgentSampler effort levels. Every level requests one candidate.}
\label{tab:effort}
\end{table}
```

Replace the old five-level formula and discussion with:

```latex
Let $h_e\in\{5,10,20\}$ be the candidate-history window for effort $e$ and
let $r_e,n_e\in\{0,1\}$ denote whether explicit reasoning and persistent notes
are enabled. Table~\ref{tab:effort} instantiates these controls. If $H_{h_e}$
forms the incumbent, promising, recent, and weak summaries from that window,
the prompt context is
\begin{equation}
  P_t^e = \left(\mathcal{X},\{c_j\}_{j=1}^{d},
  H_{h_e}(\mathcal{D}_t),r_e,n_e\right).
\end{equation}
All levels request one candidate. Medium and high share the same qualitative
prompt blocks and backend reasoning flag; high differs by retaining twice as
many candidate trials in the summary pool.
```

Change the complexity sentence to omit intermediate-report serialization:

```latex
\textbf{Complexity.} For $T$ trials and history window $h_e$, prompt construction
is $O(\min(t,h_e)d)$ at trial $t$ plus ranking a constant-size bounded window.
```

- [ ] **Step 4: Make all retained old experiment labels explicitly legacy**

Update the vision table rows to `GPT-5.5 xhigh (legacy)`, update the table captions and discussion to call the recorded sweep a legacy five-level experiment, and replace the ablation lead with:

```latex
The retained effort ablation predates the current three-level ladder in
Table~\ref{tab:effort}: it varies the legacy low--max sampler settings while
holding the model at GPT-5.5/Codex. We preserve the original labels rather than
retroactively mapping them onto the current prompts. We track
```

State that the legacy sweep rises from low to xhigh, and do not claim it directly ablates the current prompt ladder. Update Figure~`\ref{fig:cifar-effort}` to identify its right panel as the legacy five-level analytic effort ablation.

- [ ] **Step 5: Audit the method update**

Run:

```bash
rg -n 'ranked candidates|completed, failed, and pruned|medium & 15|xhigh & all|max & all|high and above|\\emph\{max\}|serialization of intermediate|GPT-5\.5 xhigh(?! \(legacy\))|from low to xhigh' paper/src/main.tex --pcre2
```

Expected: no matches. Every remaining `xhigh` or `max` occurrence must be in a sentence or label containing `legacy`.

Run:

```bash
git diff --check -- paper/src/main.tex
```

Expected: exit 0 with no output.

---

### Task 2: Add the Runtime Prompt Appendix

**Files:**
- Modify: `paper/src/main.tex:802-804`

**Interfaces:**
- Consumes: exact strings and branches from `AgentSampler._prompt`, `AgentSampler.propose`, and `AgentPruner.should_prune`.
- Produces: `Appendix~\ref{app:prompts}` with `AgentSampler` and `AgentPruner` templates.

- [ ] **Step 1: Verify that the appendix is absent**

Run:

```bash
rg -n '\\appendix|\\section\{Prompt Templates\}|app:prompts|AgentSampler Prompt|AgentPruner Prompt' paper/src/main.tex
```

Expected: only the forward `app:prompts` reference added in Task 1; no appendix section or prompt headings.

- [ ] **Step 2: Add the sampler template after the bibliography**

Immediately after `\bibliography{optim-agent}`, add `\appendix`, `\section{Prompt Templates}`, `\label{app:prompts}`, and a short statement that bracketed annotations describe substitutions and conditional blocks and are not sent to the backend.

Add `\subsection{AgentSampler Prompt}` and a manually wrapped `small`/`verbatim` block containing this complete parametric structure in the implementation's exact order:

```text
You are an expert hyperparameter-optimization engine. Think both
qualitatively (what the trend and the meaning of each parameter
suggest) and quantitatively (the numbers in the history) before
choosing the next point.

Goal: [DIRECTION] the objective value.

[If study context is present:]
What is being tuned: [STUDY CONTEXT]

[If context is present and reasoning is enabled:]
Context-derived priors:
- Prefer stable, plausible training settings before extreme
  exploration.
- For neural nets, start from moderate learning rates,
  low-to-moderate regularization/dropout, enough width/depth,
  and augmentation only when history shows it helps.
- Treat parameter names and descriptions as semantic hints,
  not just tokens.
[If context contains "early reward":]
- This run is scored by the sum of incumbent best errors, so
  early reliable improvements beat risky late exploration.

Search space:
- [NAME]: [TYPE, BOUNDS OR CHOICES, LOG SCALE, CONTEXT]
[Repeated for every active parameter.]

History summary:
[If a completed best trial exists:]
- Best trial: #[NUMBER] value=[VALUE] params=[PARAMS]
- Promising trials:
  - #[NUMBER]: value=[VALUE], params=[PARAMS]
  [Up to five, best first, from the effort window.]
- Recent trials:
  - #[NUMBER]: value=[VALUE], params=[PARAMS]
  [Up to five, oldest first.]
- Failed or weak regions to avoid:
  - #[NUMBER]: value=[VALUE], params=[PARAMS]
  [Up to three weak trials.]

[If a completed best trial exists:]
Best so far: trial [NUMBER], value=[VALUE], params=[PARAMS]

[If notes are enabled and a prior note exists:]
Your notes from previous trials: [NOTE]

Propose the next point to evaluate. Balance exploration of
unvisited regions against exploitation around promising ones;
never repeat an already-evaluated point exactly.
[If context contains "early reward":]
Because the score rewards fast incumbent-best decrease, pick a
high-confidence configuration likely to improve the best value now.
[If reasoning is enabled:]
Use the task context as priors when available: prefer choices that
make sense for the described setup unless the trial history clearly
contradicts them.
Reply with ONLY a JSON object: {"[NAME]": <value>, ...}.
[If reasoning is enabled:]
Include a short "_reasoning" field explaining your choice.
[If notes are enabled:]
Include a "_note" field: observations about the landscape worth
carrying forward to the next trial (it will be shown back to you).
```

Follow the block with the effort activation table from Task 1 and state that the history pool contains the last 5/10/20 eligible trials with the completed incumbent prepended if it falls outside the window. Clarify that despite the literal `Failed or weak` heading, failed trials are not eligible in the current implementation.

- [ ] **Step 3: Add the retry suffix**

Add a short paragraph and literal block:

```text
Your previous reply could not be parsed into valid parameters.
Reply again with ONLY the JSON object, values inside the stated
ranges.
```

State that this suffix is appended after the first invalid response and that a second invalid response triggers random fallback.

- [ ] **Step 4: Add the pruner template**

Add `\subsection{AgentPruner Prompt}` and this complete template:

```text
You are deciding whether to stop (prune) an in-progress
hyperparameter trial early. Goal: [DIRECTION] the objective.

Intermediate curves of the best completed trials (step, value):
- trial [NUMBER] (final [VALUE]): [INTERMEDIATE CURVE]
[Up to five completed trials, best first.]

Current trial params: [PARAMS]
Current trial curve so far: [INTERMEDIATE CURVE]

[LEVEL STANCE]
Reply with ONLY a JSON object: {"prune": true} or
{"prune": false}.
```

Follow it with the exact stance mapping:

```latex
\begin{itemize}
\item \emph{loose}: ``Only prune if the trial is almost certainly hopeless.''
\item \emph{medium}: ``Prune when the trial is clearly underperforming past trials.''
\item \emph{tight}: ``Prune aggressively at the first solid sign of underperformance.''
\end{itemize}
```

State that the pruner keeps the trial on an invalid response or backend failure.

- [ ] **Step 5: Verify source/template parity**

Run:

```bash
rg -n 'expert hyperparameter-optimization engine|Context-derived priors|History summary|Failed or weak regions|_reasoning|_note|previous reply could not be parsed|deciding whether to stop|Intermediate curves of the best|almost certainly hopeless|clearly underperforming|first solid sign|"prune": true' paper/src/main.tex
```

Expected: every sampler branch, retry instruction, pruner section, and stance appears in the appendix.

Compare these matches line by line with:

```bash
sed -n '85,149p' optim_agent/samplers.py
sed -n '42,52p' optim_agent/pruners.py
```

Expected: identical runtime wording apart from line wrapping, bracketed substitutions, and conditional annotations.

- [ ] **Step 6: Build and inspect the paper**

Run:

```bash
bash paper/src/verify_paper.sh
```

Expected: exit 0 and a numeric score. The total page count may exceed the verifier's nine-page submission cap because the requested appendix is post-reference; the main text must still end before references on page 8.

Run:

```bash
rg -n 'Undefined control sequence|LaTeX Error|Overfull \\hbox|Output written' paper/src/main.log
rg -n 'sec:references-start|app:prompts' paper/src/main.aux
```

Expected: no undefined-control-sequence or LaTeX-error matches; no overfull boxes; one `Output written` line; both labels resolved.

Run:

```bash
git diff --check -- paper/src/main.tex
git diff --stat -- paper/src/main.tex
git status --short
```

Expected: clean diff check, only `paper/src/main.tex` changed by this implementation, and all unrelated pre-existing worktree changes preserved.

