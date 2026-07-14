# Claude And Codex Plugin Wrappers Design

## Goal

Make optim-agent installable as a Claude and Codex plugin from the canonical
GitHub repository without duplicating the root `SKILL.md` workflow.

## Shared Metadata

- plugin name: `optim-agent`
- version: `0.1.1`
- developer: `Optim-Agent`
- repository: `https://github.com/Optim-Agent/optim-agent`
- homepage: `https://optim-agent.github.io/optim-agent/`
- license: `MIT`

## Claude Wrapper

Add:

```text
.claude-plugin/plugin.json
.claude-plugin/marketplace.json
```

Claude reads the existing repository-root `SKILL.md`. The plugin manifest does
not declare a copied skills directory. The marketplace contains one available
plugin named `optim-agent` whose source is the repository root.

Expected installation:

```bash
claude plugin marketplace add Optim-Agent/optim-agent
claude plugin install optim-agent@optim-agent
```

## Codex Wrapper

Add:

```text
.codex-plugin/plugin.json
.agents/plugins/marketplace.json
skills/optim-agent/SKILL.md
```

The Codex manifest points `skills` to `./skills/`. The nested Skill is a thin
forwarder with valid frontmatter that tells the agent to read and follow the
canonical `../../SKILL.md`; it does not repeat the workflow.

The Codex marketplace contains one available plugin named `optim-agent` whose
source is the repository root.

## Documentation

Add concise Claude and Codex plugin installation commands to `README.md` while
keeping the existing PyPI and direct Skill installation paths.

## Validation

- Validate the Codex plugin with the bundled plugin validator.
- Parse all four JSON manifests in tests.
- Assert both plugin manifests use version `0.1.1` and developer `Optim-Agent`.
- Assert both marketplace manifests expose one `optim-agent` entry pointing to
  the repository root.
- Assert the Codex forwarder references `../../SKILL.md` and does not contain a
  copy of the root workflow.
- Run the full repository test suite and `git diff --check`.

## Constraints

- Keep root `SKILL.md` as the only complete workflow source.
- Do not use symlinks.
- Do not add hooks, MCP servers, apps, scripts, assets, credentials, or runtime
  dependencies to either plugin.
- Do not change package or benchmark behavior.
