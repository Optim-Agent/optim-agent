# Paper Novelty Framing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make optim-agent's semantics-aware HPO systems contribution explicit and mathematically precise while preserving all empirical caveats and the seven-page AAAI limit.

**Architecture:** Revise existing prose rather than add a novelty section or comparison table. Establish the claim in the title and abstract, differentiate it from prior LLM-HPO work in the introduction and related work, formalize semantic blindness in the method, and carry the distinction through discussion and conclusion.

**Tech Stack:** AAAI 2027 LaTeX, pdfTeX, BibTeX, macOS PDFKit rendering, repository paper verifier.

---

### Task 1: Make the Novelty Legible Upfront

**Files:**
- Modify: `paper/src/main.tex:21-170`

- [ ] **Step 1: Replace the title**

Use:

```latex
\title{Optim-Agent: Validated Semantics-Aware Hyperparameter Optimization with CLI Agents}
```

- [ ] **Step 2: Replace the abstract with a semantics-first abstract**

The replacement must contain these exact claims in cohesive prose:

```latex
Conventional hyperparameter optimization maps typed coordinates and numerical
histories to candidates without representing what each coordinate means. We
present \emph{optim-agent}
(\url{https://github.com/Optim-Agent/optim-agent}), a semantics-aware HPO
interface that exposes an existing command-line LLM agent as a sampler and
optional pruner through an ask--tell study API. Study and parameter descriptions
become first-class sampler inputs alongside distributions and trial history.
The agent proposes JSON, while the package validates every value, retries once,
and otherwise samples a valid random vector; the objective and declared search
space remain authoritative. This combination brings semantic priors into an
ordinary HPO workflow without training a meta-optimizer or delegating the
training loop to an autonomous agent. Studies persist to JSON or SQLite, and
the core package uses only Python's standard library.
```

Follow this contribution statement with the existing MNIST, CIFAR-10, analytic,
and credit-default findings and their current caveats. Preserve the repository
URL, all reported numbers, the startup-confound disclosure, the non-significant
credit/TPE result, and the statement that the evidence establishes feasibility
rather than general superiority.

- [ ] **Step 3: Rewrite the introduction's problem gap and package boundary**

Replace the current generic semantic-prior framing with this distinction:

```latex
The missing input is not another numerical observation but the meaning of a
coordinate. Given the same bounds and outcomes, a context-blind sampler treats
``dropout'' and ``crop padding'' only as differently named dimensions;
optim-agent can condition on their roles in regularization and augmentation.
The proposal policy changes, but the objective, budget, legal search space, and
study state remain under the user's control.
```

Describe the package as the reusable boundary joining four existing ideas:
semantic schemas, already-authenticated CLI agents, conventional define-by-run
and ask--tell HPO, and deterministic validation/fallback with persistent records.
Do not call contextual candidate generation itself new.

- [ ] **Step 4: Replace the contribution bullets**

Use three bullets with these scopes:

```latex
\begin{itemize}
\item a semantics-aware sampler contract for mixed HPO spaces, where study and
parameter descriptions augment numerical trial history without changing the
objective or legal domain;
\item a dependency-light implementation that reuses interchangeable CLI agents
behind a familiar ask--tell/define-by-run API, typed validation, bounded retry
and fallback, and JSON or SQLite storage;
\item an auditable small-budget evaluation that reports gains on analytic and
image objectives, validation-separated tabular results, statistical tests, and
the confounds that prevent a semantic-only causal claim.
\end{itemize}
```

- [ ] **Step 5: Sharpen the closest-work comparison**

Retain all current citations and the acknowledgement that LLM-based HPO is not
new. Replace the concluding comparison with:

```latex
The distinction is the systems boundary. OptFormer trains a trajectory model;
LLAMBO inserts LLM components into Bayesian optimization; AgentHPO and
AutoML-Agent expand agent control across an experiment pipeline; and OPRO uses
prompted optimization as the task. optim-agent instead makes an existing CLI
agent a replaceable sampler inside an ordinary user-owned HPO study. Typed
distributions constrain its action, deterministic validation governs execution,
and persistent records preserve the context--proposal--outcome trace. The
contribution is this reusable composition, not the first use of an LLM to
suggest hyperparameters.
```

- [ ] **Step 6: Compile the first manuscript iteration**

Run:

```bash
cd paper/src
./verify_paper.sh
```

Expected: pdfTeX completes. Record the verifier score and actual references
page before making any space adjustment.

### Task 2: Formalize and Carry the Distinction Through the Paper

**Files:**
- Modify: `paper/src/main.tex:175-345`
- Modify: `paper/src/main.tex:620-720`

- [ ] **Step 1: Define semantic blindness in the method**

After defining the numerical history, define
$C=(c_{\mathrm{study}},\{c_j\}_{j=1}^{d})$. Add this unnumbered display:

```latex
\[
p_A(x_{t+1}\mid\mathcal X,\mathcal D_t,C)
=p_A(x_{t+1}\mid\mathcal X,\mathcal D_t,C')
\quad\text{for all }C,C'.
\]
```

State that this is semantic blindness, not a defect: random search, TPE, and
other numerical policies intentionally satisfy it when descriptions are not
part of their input. Then define optim-agent's proposal as generally dependent
on $C$:

```latex
\begin{equation}
  \tilde{x}_{t+1}\sim\pi_\theta(
  \mathcal X,C,\mathcal D_t,\mathrm{direction},h,r,n,e),
\end{equation}
```

Define $h,r,n,e$ consistently with the existing independent-controls paragraph.
Do not create another numbered equation for semantic blindness.

- [ ] **Step 2: Connect validation to the novelty**

End `Validation as partial projection` with:

```latex
Semantic context may therefore change which point is proposed, but it cannot
redefine a bound, introduce an undeclared category, or bypass the objective.
This separation between a language-conditioned proposal and mechanically
validated execution is the central interface boundary.
```

Delete an equivalent repeated explanation elsewhere in `AgentSampler` so the
method does not grow materially.

- [ ] **Step 3: Strengthen the discussion without claiming explainability**

Replace repeated package-interface prose with two compact points:

```latex
Semantic conditioning is useful only when parameter meanings carry priors that
are informative relative to the small trial budget. It can also encode stale or
incorrect conventions; numerical outcomes must therefore remain authoritative,
and random and TPE remain necessary controls.

Persistent records pair the declared semantics, proposal, and observed outcome,
creating an auditable semantic trace. Optional rationales and notes can help a
human inspect the process, but they are model-generated text rather than
faithful causal explanations of the proposal.
```

Retain the existing limits on external validity, startup confounding, five-seed
power, and credit deployment.

- [ ] **Step 4: Replace the conclusion**

Use:

```latex
\emph{optim-agent} makes parameter semantics a first-class input to an ordinary
HPO study while keeping CLI agents behind typed validation, bounded fallback,
and persistent records. This reusable boundary, rather than LLM-based HPO
itself, is the contribution. The results show promising small-budget behavior,
while the matched validation study and disclosed confounds do not establish
general superiority, faithful explanation, or an isolated context effect.
```

- [ ] **Step 5: Scan for contradictory novelty claims**

Run:

```bash
grep -n -i -E 'first|only contextual|explainable|faithful|robust|general superiority|semantic-only' paper/src/main.tex
```

Expected: every match is either a qualified limitation, an explicit denial of
an unsupported claim, or ordinary prose unrelated to novelty.

- [ ] **Step 6: Compile the second manuscript iteration**

Run:

```bash
cd paper/src
./verify_paper.sh
```

Expected: score `100`, nine total pages, and `sec:references-start` on page 8.
If main content reaches page 8, recover space only by deleting repeated prose in
Related Work or Discussion. Do not change type size, margins, spacing, or figure
dimensions.

### Task 3: Verify and Deliver the PDF

**Files:**
- Verify: `paper/src/main.tex`
- Verify: `paper/src/main.log`
- Generate: `paper/src/main.pdf`
- Generate: `paper/main.pdf`

- [ ] **Step 1: Run a clean final pdfTeX verification**

Run:

```bash
cd paper/src
./verify_paper.sh
grep -E 'Output written|Overfull|LaTeX Warning|undefined|multiply defined' main.log
grep -n 'sec:references-start' main.aux
```

Expected: verifier `100`; nine pages; no overfull boxes, undefined citations, or
undefined references; references label on page 8.

- [ ] **Step 2: Verify novelty markers and preserved evidence**

Run:

```bash
grep -n -E 'Semantics-Aware|semantic blindness|first-class input|systems boundary|auditable semantic trace' main.tex
grep -o 'https://github.com/Optim-Agent/optim-agent' main.tex | wc -l
grep -n -E '38\.2|20\.8|p=\.055|p=\.925|p=\.201|p=\.205' main.tex
```

Expected: the novelty markers appear in the intended sections; the repository
URL count is `2`; all empirical values remain present.

- [ ] **Step 3: Render and visually inspect all pages**

Run:

```bash
rm -rf tmp/pdfs
mkdir -p tmp/pdfs
swift /tmp/render_optim_agent_pdf.swift paper/src/main.pdf tmp/pdfs
swift /tmp/check_optim_agent_reference_boundary.swift paper/src/main.pdf
```

Expected: nine PNG pages; page 8 begins with `References`. Inspect every page
for clipping, overlapping floats, broken equations, unreadable tables, and
column-transition defects. Inspect pages 1--3 specifically for the revised
novelty argument and page 7 for a substantive conclusion before the boundary.

- [ ] **Step 4: Copy the verified artifact**

Run:

```bash
cp paper/src/main.pdf paper/main.pdf
cmp -s paper/src/main.pdf paper/main.pdf
```

Expected: `cmp` exits `0`.

- [ ] **Step 5: Leave unrelated work untouched**

Run:

```bash
git status --short
```

Expected: existing RL-control artifacts and user changes remain present and
unmodified. The manuscript remains under the repository's intentional
`/paper/` ignore rule, so no paper commit is created.
