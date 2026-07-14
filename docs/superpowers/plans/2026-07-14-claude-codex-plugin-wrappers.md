# Claude And Codex Plugin Wrappers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repository installable as an Optim-Agent Claude Code plugin and Codex plugin while keeping the root `SKILL.md` as the only complete workflow source.

**Architecture:** Each marketplace points to one standard plugin directory at `plugins/optim-agent`. Claude discovers a root-level thin Skill forwarder; Codex discovers a nested thin Skill forwarder through its manifest. Both forwarders direct the active agent to the canonical repository-root `SKILL.md` without copying its workflow.

**Tech Stack:** JSON plugin manifests, Markdown Agent Skills, Python standard-library JSON/path tests, Claude Code CLI, Codex bundled plugin validator, pytest.

---

### Task 1: Specify the wrapper contract in tests

**Files:**
- Create: `tests/test_plugin_manifests.py`

- [ ] **Step 1: Add the failing manifest and forwarder tests**

Create `tests/test_plugin_manifests.py` with helpers that load repository-relative JSON and these assertions:

```python
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "plugins" / "optim-agent"


def _json(relative_path):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_plugin_manifests_share_public_metadata():
    claude = _json("plugins/optim-agent/.claude-plugin/plugin.json")
    codex = _json("plugins/optim-agent/.codex-plugin/plugin.json")

    assert claude["name"] == codex["name"] == "optim-agent"
    assert claude["version"] == codex["version"] == "0.1.1"
    assert claude["author"]["name"] == codex["author"]["name"] == "Optim-Agent"
    assert claude["repository"] == codex["repository"] == (
        "https://github.com/Optim-Agent/optim-agent"
    )
    assert claude["homepage"] == codex["homepage"] == (
        "https://optim-agent.github.io/optim-agent/"
    )
    assert claude["license"] == codex["license"] == "MIT"
    assert codex["skills"] == "./skills/"


def test_marketplaces_expose_the_same_single_plugin():
    claude = _json(".claude-plugin/marketplace.json")
    codex = _json(".agents/plugins/marketplace.json")

    for marketplace in (claude, codex):
        assert len(marketplace["plugins"]) == 1
        plugin = marketplace["plugins"][0]
        assert plugin["name"] == "optim-agent"
        source = plugin["source"]
        path = source["path"] if isinstance(source, dict) else source
        assert path == "./plugins/optim-agent"


def test_plugin_skills_forward_to_the_canonical_workflow():
    canonical = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    claude = (PLUGIN / "SKILL.md").read_text(encoding="utf-8")
    codex = (PLUGIN / "skills" / "optim-agent" / "SKILL.md").read_text(
        encoding="utf-8"
    )

    assert "../../SKILL.md" in claude
    assert "../../../../SKILL.md" in codex
    for forwarder in (claude, codex):
        assert len(forwarder) < 800
        assert "## Workflow" not in forwarder
        assert "## Recovery" not in forwarder
        assert forwarder != canonical
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_plugin_manifests.py -q
```

Expected: FAIL because the plugin manifests and forwarders do not exist.

- [ ] **Step 3: Commit the tests**

```bash
git add tests/test_plugin_manifests.py
git commit -m "Test Claude and Codex plugin packaging"
```

### Task 2: Add the standard plugin and marketplace layouts

**Files:**
- Create: `.claude-plugin/marketplace.json`
- Create: `.agents/plugins/marketplace.json`
- Create: `plugins/optim-agent/.claude-plugin/plugin.json`
- Create: `plugins/optim-agent/.codex-plugin/plugin.json`
- Create: `plugins/optim-agent/SKILL.md`
- Create: `plugins/optim-agent/skills/optim-agent/SKILL.md`

- [ ] **Step 1: Add the Claude marketplace and manifest**

Use one marketplace entry named `optim-agent` with source `./plugins/optim-agent`. Use shared version `0.1.1`, Optim-Agent authorship, repository/homepage/license metadata, and only the keywords `optimization`, `hpo`, and `agent-skills`.

- [ ] **Step 2: Add the Codex marketplace and manifest**

Use a local Codex marketplace source with path `./plugins/optim-agent`, installation policy `AVAILABLE`, authentication policy `ON_INSTALL`, and category `Productivity`. Point the Codex plugin's `skills` field to `./skills/`; keep capabilities empty and use this default prompt:

```text
Optimize this system's configurable parameters against its measurable objective.
```

- [ ] **Step 3: Add two thin Skill forwarders**

Use valid `name` and `description` frontmatter in each file. The Claude forwarder at `plugins/optim-agent/SKILL.md` must instruct the agent to read and follow `../../SKILL.md`. The Codex forwarder at `plugins/optim-agent/skills/optim-agent/SKILL.md` must instruct the agent to read and follow `../../../../SKILL.md`. Do not copy any workflow section from the canonical Skill.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
python3 -m pytest tests/test_plugin_manifests.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit the packaging**

```bash
git add .claude-plugin .agents/plugins plugins/optim-agent
git commit -m "Add Claude and Codex plugin wrappers"
```

### Task 3: Document plugin installation

**Files:**
- Modify: `README.md`
- Modify: `tests/test_plugin_manifests.py`

- [ ] **Step 1: Add a failing README assertion**

Add a test that requires these exact command pairs in `README.md`:

```text
claude plugin marketplace add Optim-Agent/optim-agent
claude plugin install optim-agent@optim-agent
codex plugin marketplace add Optim-Agent/optim-agent
codex plugin add optim-agent@optim-agent
```

- [ ] **Step 2: Run the README test and verify RED**

Run the focused test and expect failure because the commands are absent.

- [ ] **Step 3: Add concise install commands to Skill Mode**

Keep the existing direct `$skill-installer` command and add separate Claude Code and Codex command blocks immediately after it. Do not remove PyPI, GitHub source, skills.sh, AgentSkill.sh, or ClawHub installation documentation elsewhere in the README.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
python3 -m pytest tests/test_plugin_manifests.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit the documentation**

```bash
git add README.md tests/test_plugin_manifests.py
git commit -m "Document Claude and Codex plugin installs"
```

### Task 4: Validate installability and repository health

**Files:**
- Modify only if a validator identifies a concrete schema error.

- [ ] **Step 1: Validate both Claude manifests**

Run:

```bash
claude plugin validate --strict plugins/optim-agent
claude plugin validate --strict .
```

Expected: both validations succeed without errors.

- [ ] **Step 2: Validate the Codex plugin manifest**

Run:

```bash
python3 /Users/zhuofan/.codex/skills/.system/plugin-creator/scripts/validate_plugin.py plugins/optim-agent
```

Expected: validator reports a valid plugin.

- [ ] **Step 3: Test marketplace installation in isolated homes**

Create empty temporary Claude and Codex homes, add this repository as each marketplace, install `optim-agent@optim-agent`, and confirm the installed plugin is listed. The temporary homes must remain outside the repository.

- [ ] **Step 4: Run full verification**

Run:

```bash
python3 -m pytest -q
git diff --check
git status --short
```

Expected: all tests pass, `git diff --check` is silent, and only intentional uncommitted validator fixes are listed.

- [ ] **Step 5: Commit validator fixes if needed**

If validation required manifest corrections, stage only those corrections and commit them as:

```bash
git commit -m "Fix plugin manifest validation"
```
