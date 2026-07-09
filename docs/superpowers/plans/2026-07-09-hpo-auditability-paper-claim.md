# HPO Auditability Paper Claim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact Discussion paragraph explaining that agent-based HPO improves auditability for high-stakes, risk-managed ML workflows.

**Architecture:** This is a paper-text-only change. Add one paragraph immediately after the existing Discussion paragraph that ends with "produce the same study records." Keep the claim cautious and domain-general.

**Tech Stack:** LaTeX, AAAI 2027 template, existing `verify_paper.sh`.

## Global Constraints

- Edit only `paper/src/main.tex`.
- Do not add a new section, experiment, figure, or explicit finance/quant example.
- Avoid "only way"; use cautious wording such as "distinctive advantage" or "practical route."
- Run `./verify_paper.sh` from `paper/src` after the edit.

---

### Task 1: Add Auditability Value Claim

**Files:**
- Modify: `paper/src/main.tex`

**Interfaces:**
- Consumes: existing Discussion text around the paragraph ending in "produce the same study records."
- Produces: one new Discussion paragraph suitable for AAAI main content.

- [ ] **Step 1: Inspect target context**

Run:

```bash
nl -ba paper/src/main.tex | sed -n '668,686p'
```

Expected: output includes the paragraph about validated suggestions, auditable agent behavior, and shared study records.

- [ ] **Step 2: Insert the paragraph**

Add this paragraph after the sentence ending `produce the same study records.`:

```tex
A distinctive advantage of this interface is that it can make HPO decisions
more explainable than purely empirical samplers. The recorded proposal can be
paired with parameter meanings, observed trial history, and optional agent
rationales, so a team can inspect why a configuration was plausible before it
was evaluated. This is especially valuable in high-stakes, risk-managed ML
workflows, where a good hyperparameter set is not enough: the path by which it
was selected must also be auditable.
```

- [ ] **Step 3: Verify the paper**

Run:

```bash
cd paper/src && ./verify_paper.sh
```

Expected: final output is `100`.

- [ ] **Step 4: Check page placement**

Run:

```bash
cd paper/src && python3 - <<'PY'
from pathlib import Path
import re
log = Path("main.log").read_text(errors="ignore")
aux = Path("main.aux").read_text(errors="ignore")
pages = re.search(r"Output written on .*?\((\d+) pages?", log)
ref = re.search(r"\\newlabel\{sec:references-start\}\{\{.*?\}\{(\d+)\}", aux)
print("pages", pages.group(1) if pages else "unknown")
print("references_start_page", ref.group(1) if ref else "unknown")
PY
```

Expected: `pages 9` and `references_start_page 8`.

- [ ] **Step 5: Commit the paper edit**

Run:

```bash
git add -f paper/src/main.tex
git commit -m "paper: emphasize HPO auditability"
```

Expected: only `paper/src/main.tex` is committed.
