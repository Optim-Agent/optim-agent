# README Practical Values Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact `README.md` section that summarizes the paper's core practical value for users.

**Architecture:** This is a Markdown-only documentation change. Insert one section after the existing `## Why optim-agent` bullet list and before the full documentation link, preserving the rest of the README.

**Tech Stack:** Markdown, existing README assets and links.

## Global Constraints

- Modify only `README.md`.
- Preserve the current README structure and benchmark sections.
- Do not rewrite the opening, change benchmark numbers, or add new experiments.
- Keep the new section compact.
- No code tests are needed for this README-only copy change.

---

### Task 1: Add Practical Values Section

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: existing `## Why optim-agent` section.
- Produces: a new `## What you get in practice` section before the full documentation link.

- [ ] **Step 1: Inspect target context**

Run:

```bash
nl -ba README.md | sed -n '24,58p'
```

Expected: output includes the `## Why optim-agent` heading, its bullets, and the `Full documentation` line.

- [ ] **Step 2: Insert the new section**

Add this Markdown block after the `Zero runtime dependencies` bullet and before `**Full documentation:**`:

```markdown
## What you get in practice

- **Small-budget leverage** — useful when each trial is expensive and classical
  surrogates are still data-starved.
- **Semantic tuning** — combines parameter meanings, study context, and trial
  history instead of treating every knob as an anonymous coordinate.
- **Auditable decisions** — records the context, proposals, outcomes, and
  optional agent rationales that matter in high-stakes, risk-managed ML
  workflows.
- **Bounded autonomy** — the agent proposes, optim-agent validates, and your
  objective decides; invalid agent output falls back to safe sampling.
- **Drop-in deployment** — Optuna-style API, JSON/SQLite storage, local
  authenticated CLI backends, and zero runtime dependencies.
```

- [ ] **Step 3: Review edited Markdown in context**

Run:

```bash
sed -n '20,80p' README.md
```

Expected: the new section appears once, sits between `## Why optim-agent` and `**Full documentation:**`, and keeps line wrapping readable.

- [ ] **Step 4: Check the scoped diff**

Run:

```bash
git diff -- README.md | sed -n '1,180p'
```

Expected: the diff only adds the compact `## What you get in practice` section.

- [ ] **Step 5: Commit the README edit**

Run:

```bash
git add README.md
git commit -m "docs: add README practical values"
```

Expected: `README.md` is committed.
