# Skill Hub Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish optim-agent 0.1.1 under the Optim-Agent brand on skills.sh, AgentSkill.sh, and ClawHub without duplicating `SKILL.md` in the repository.

**Architecture:** Keep GitHub as the only source of truth. Use repository-based discovery for skills.sh and AgentSkill.sh, and publish a temporary one-file package to ClawHub with pinned GitHub provenance metadata.

**Tech Stack:** GitHub, root `SKILL.md`, skills CLI, AgentSkill.sh web submission, ClawHub CLI.

---

### Task 1: Validate the canonical skill

**Files:**
- Read: `SKILL.md`

- [ ] **Step 1: Validate repository discovery**

Run:

```bash
NPM_CONFIG_CACHE=/tmp/optim-agent-npm-cache \
  npx --yes skills add Optim-Agent/optim-agent --list
```

Expected: exactly one available skill named `optim-agent`.

- [ ] **Step 2: Confirm the pinned source file**

Run:

```bash
git show 5206d0e:SKILL.md >/tmp/optim-agent-SKILL.md
cmp SKILL.md /tmp/optim-agent-SKILL.md
```

Expected: no output; the current skill matches the pinned source commit.

### Task 2: Register and verify skills.sh

**Files:**
- Temporary: `/tmp/optim-agent-skills-sh/`

- [ ] **Step 1: Install from GitHub in a temporary project**

Run:

```bash
mkdir -p /tmp/optim-agent-skills-sh
cd /tmp/optim-agent-skills-sh
NPM_CONFIG_CACHE=/tmp/optim-agent-npm-cache \
  npx --yes skills add Optim-Agent/optim-agent \
  --skill optim-agent --agent codex --copy -y
```

Expected: installation succeeds and records the repository installation used by skills.sh discovery.

- [ ] **Step 2: Verify the installed skill**

Run:

```bash
NPM_CONFIG_CACHE=/tmp/optim-agent-npm-cache npx --yes skills list --json
```

Expected: JSON includes `optim-agent` sourced from `Optim-Agent/optim-agent`.

### Task 3: Submit AgentSkill.sh

**Files:** None.

- [ ] **Step 1: Open the submission page**

Navigate to `https://agentskill.sh/submit` using the signed-in browser session.

- [ ] **Step 2: Submit the canonical repository**

Submit `https://github.com/Optim-Agent/optim-agent` using the `Optim-Agent` identity where the site requests attribution. Do not upload a copied file or archive.

- [ ] **Step 3: Verify the resulting entry**

Confirm that the entry name is `optim-agent`, its source resolves to the canonical GitHub repository, and its description comes from root `SKILL.md`.

### Task 4: Publish ClawHub

**Files:**
- Temporary: `/tmp/optim-agent-clawhub/SKILL.md`

- [ ] **Step 1: Authenticate**

Run:

```bash
NPM_CONFIG_CACHE=/tmp/optim-agent-npm-cache npx --yes clawhub login
NPM_CONFIG_CACHE=/tmp/optim-agent-npm-cache npx --yes clawhub whoami
```

Expected: authenticated account is authorized to use the `Optim-Agent` publisher.

- [ ] **Step 2: Ensure the publisher exists**

Inspect `clawhub publisher` commands. Create or select `Optim-Agent` only if the authenticated account is authorized; do not fall back to a personal publisher.

- [ ] **Step 3: Build a one-file temporary payload**

Run:

```bash
rm -rf /tmp/optim-agent-clawhub
mkdir -p /tmp/optim-agent-clawhub
cp SKILL.md /tmp/optim-agent-clawhub/SKILL.md
```

- [ ] **Step 4: Dry-run publication**

Run:

```bash
NPM_CONFIG_CACHE=/tmp/optim-agent-npm-cache npx --yes clawhub publish \
  /tmp/optim-agent-clawhub \
  --slug optim-agent --name optim-agent --owner Optim-Agent \
  --version 0.1.1 --tags latest,hpo,optimization \
  --topics bayesian-optimization,hyperparameter-optimization,agent-skills \
  --source-repo https://github.com/Optim-Agent/optim-agent \
  --source-commit 5206d0e --source-ref main --source-path . \
  --dry-run --json
```

Expected: `would-publish`, version `0.1.1`, and one packaged file.

- [ ] **Step 5: Publish**

Repeat the command without `--dry-run` after the payload and publisher checks pass.

### Task 5: Verify all three distributions

**Files:** None.

- [ ] **Step 1: Verify ClawHub search and metadata**

Run `clawhub search optim-agent`, `clawhub inspect optim-agent`, and install it into a temporary directory. Confirm version `0.1.1`, publisher `Optim-Agent`, canonical source metadata, and a valid `SKILL.md`.

- [ ] **Step 2: Recheck skills.sh and AgentSkill.sh**

Confirm both entries resolve to `Optim-Agent/optim-agent`. If indexing is asynchronous, record the successful submission/installation and report the pending index state rather than resubmitting duplicates.

- [ ] **Step 3: Check repository cleanliness**

Run:

```bash
git status --short
```

Expected: no Hub credentials, generated packages, or temporary files in the repository.
