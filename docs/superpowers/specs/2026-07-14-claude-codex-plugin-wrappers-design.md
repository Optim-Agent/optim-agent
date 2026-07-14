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

## Marketplace Layout

Add:

```text
.claude-plugin/marketplace.json
.agents/plugins/marketplace.json
plugins/optim-agent/
├── .claude-plugin/plugin.json
├── .codex-plugin/plugin.json
├── SKILL.md
└── skills/
    └── optim-agent/
        └── SKILL.md
```

Both marketplace manifests contain one available plugin named `optim-agent`
whose source is `./plugins/optim-agent`. This standard marketplace layout is
required because Codex does not discover a plugin when a marketplace entry
points to the repository root.

## Claude Wrapper

The Claude plugin manifest lives under `plugins/optim-agent/.claude-plugin/`.
Claude loads `plugins/optim-agent/SKILL.md`, a thin forwarder with valid
frontmatter that tells the agent to read and follow the canonical
`../../SKILL.md`.

Expected installation:

```bash
claude plugin marketplace add Optim-Agent/optim-agent
claude plugin install optim-agent@optim-agent
```

## Codex Wrapper

The Codex manifest points `skills` to `./skills/`. Its nested Skill is a thin
forwarder with valid frontmatter that tells the agent to read and follow the
canonical `../../../../SKILL.md`; it does not repeat the workflow.

Expected installation:

```bash
codex plugin marketplace add Optim-Agent/optim-agent
codex plugin add optim-agent@optim-agent
```

## Documentation

Add concise Claude and Codex plugin installation commands to `README.md` while
keeping the existing PyPI and direct Skill installation paths.

## Validation

- Validate the Codex plugin with the bundled plugin validator.
- Parse all four JSON manifests in tests.
- Assert both plugin manifests use version `0.1.1` and developer `Optim-Agent`.
- Assert both marketplace manifests expose one `optim-agent` entry pointing to
  the repository root.
- Assert the Claude and Codex forwarders reference the canonical root
  `SKILL.md` and do not contain copies of the root workflow.
- Install both marketplace layouts from an isolated temporary checkout.
- Run the full repository test suite and `git diff --check`.

## Constraints

- Keep root `SKILL.md` as the only complete workflow source.
- Do not use symlinks.
- Do not add hooks, MCP servers, apps, scripts, assets, credentials, or runtime
  dependencies to either plugin.
- Do not change package or benchmark behavior.
