# Paper Overview Diagram Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an original, full-width comparison figure that makes optim-agent's semantics-aware and mechanically validated HPO boundary clear while preserving seven pages of main content.

**Architecture:** A self-contained TikZ document owns the editable artwork and compiles to a vector PDF included by the AAAI manuscript. The manuscript replaces the existing analytic-curves float with the new overview after the Introduction contributions, retaining the analytic table and prose.

**Tech Stack:** pdfTeX, TikZ, AAAI 2027 LaTeX template, PDFKit rendering helpers.

---

### Task 1: Build the standalone vector figure

**Files:**
- Create: `paper/src/figures/optim_agent_overview.tex`
- Create: `paper/src/figures/optim_agent_overview.pdf`

- [ ] **Step 1: Confirm the asset is initially absent**

Run: `test ! -e paper/src/figures/optim_agent_overview.tex`

Expected: exit status 0 before implementation.

- [ ] **Step 2: Create the zero-margin TikZ source**

Use an explicit `750pt` by `220pt` page, a matching TikZ bounding box, native TeX math labels, and reusable styles for charcoal, semantic-orange, proposal-blue, outcome-green, and rejected-gray/red elements. Draw three equal panels with these exact flows:

```text
Human tuning: semantic knowledge -> human expert -> manual proposal
              -> experiment -> observation -> human expert
Context-blind: typed space/history -> Random or TPE -> candidate
               -> objective -> numerical history -> Random or TPE
               semantic schema -x-> sampler (not consumed)
optim-agent: C + X + D_t -> CLI agent -> JSON proposal -> validator V
             -> user objective f(x) -> JSON/SQLite history -> next trial
             invalid -> retry once -> valid random fallback
```

The source must use only `geometry`, `tikz`, and TikZ's `arrows.meta` and `calc` libraries so it compiles in the repository's current TinyTeX installation.

- [ ] **Step 3: Compile the vector asset**

Run:

```bash
cd paper/src/figures
pdflatex -interaction=nonstopmode -halt-on-error optim_agent_overview.tex
```

Expected: exit status 0 and `optim_agent_overview.pdf` with one page.

- [ ] **Step 4: Render and inspect the figure**

Run:

```bash
mkdir -p /tmp/pdfs/optim-agent-overview
swift /tmp/render_optim_agent_pdf.swift \
  paper/src/figures/optim_agent_overview.pdf \
  /tmp/pdfs/optim-agent-overview
```

Expected: one PNG with no clipped labels, overlapping arrows, unreadable text, or copied AgentHPO artwork.

### Task 2: Integrate the overview as Figure 1

**Files:**
- Modify: `paper/src/main.tex`

- [ ] **Step 1: Add the overview after the contribution list**

Insert this exact float before `\section{Related Work}`:

```tex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/optim_agent_overview.pdf}
\caption{Three HPO boundaries. Human tuning uses semantics but requires manual
proposals. Random and TPE consume the typed space and numerical history but not
parameter descriptions. \emph{optim-agent} makes descriptions first-class
inputs to a replaceable CLI-agent proposal policy, then mechanically validates
the JSON candidate before the user objective executes it; outcomes persist to
the next trial, and invalid replies use bounded retry and valid random fallback.}
\label{fig:overview}
\end{figure*}
```

- [ ] **Step 2: Remove the displaced analytic-curves float**

Delete only the `figure*` block labeled `fig:hard`; retain Table `tab:hard` and all analytic benchmark discussion.

- [ ] **Step 3: Compile after the manuscript iteration**

Run:

```bash
cd paper/src
./verify_paper.sh
cp main.pdf ../main.pdf
```

Expected: verifier score `100`, nine total pages, and the bibliography label on page 8.

### Task 3: Verify pagination and visual quality

**Files:**
- Verify: `paper/src/main.pdf`
- Verify: `paper/main.pdf`

- [ ] **Step 1: Check the hard bibliography boundary**

Run: `swift /tmp/check_optim_agent_reference_boundary.swift paper/src/main.pdf`

Expected: `page_8_first_line=References` and exit status 0.

- [ ] **Step 2: Reject TeX layout and reference warnings**

Run:

```bash
! grep -E "Overfull \\hbox|Undefined control sequence|LaTeX Error|undefined references|Citation.*undefined" paper/src/main.log
```

Expected: exit status 0 and no matching output.

- [ ] **Step 3: Render every paper page**

Run:

```bash
rm -rf /tmp/pdfs/optim-agent-paper
mkdir -p /tmp/pdfs/optim-agent-paper
swift /tmp/render_optim_agent_pdf.swift \
  paper/src/main.pdf \
  /tmp/pdfs/optim-agent-paper
```

Expected: nine page PNGs. Inspect all pages, with particular attention to Figure 1, float ordering, column collisions, and the start of References on page 8.

- [ ] **Step 4: Confirm the delivered PDF matches the verified build**

Run: `cmp paper/src/main.pdf paper/main.pdf`

Expected: exit status 0.
