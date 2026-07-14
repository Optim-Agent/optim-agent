# Skill Hub Distribution Design

## Goal

Publish optim-agent to skills.sh, AgentSkill.sh, and ClawHub under the
`Optim-Agent` brand without adding a second copy of `SKILL.md` to the repository.

## Source Of Truth

- GitHub repository: `Optim-Agent/optim-agent`
- Skill file: repository-root `SKILL.md`
- Release version: `0.1.1`
- Source commit: `5206d0e`
- Publisher identity: `Optim-Agent`

The repository remains the only editable source. Hub entries reference the
GitHub repository and pinned source commit.

## Distribution

### skills.sh

Use the existing root `SKILL.md`, which the skills CLI already detects as the
single `optim-agent` skill. Trigger indexing through a temporary installation,
then verify repository discovery and the documented install command.

### AgentSkill.sh

Submit `https://github.com/Optim-Agent/optim-agent` through the site submission
flow. Reuse the repository name, description, homepage, and root `SKILL.md`;
do not upload a separate archive.

### ClawHub

Authenticate with the account authorized to publish under `Optim-Agent`.
Publish from a temporary directory containing only the current root
`SKILL.md`, while recording the GitHub repository, commit, ref, and source path
as provenance metadata.

Publish with:

- slug: `optim-agent`
- display name: `optim-agent`
- version: `0.1.1`
- tags: `latest,hpo,optimization`
- topics: `bayesian-optimization,hyperparameter-optimization,agent-skills`

If the `Optim-Agent` publisher does not exist or the authenticated account
cannot use it, create or request that publisher rather than publishing under a
personal namespace.

## Verification

- skills.sh: repository lists exactly one skill and a temporary install works.
- AgentSkill.sh: the submitted entry resolves to the canonical GitHub repo.
- ClawHub: search and inspect return `optim-agent@0.1.1`, the package contains
  only the intended skill payload, and a temporary installation succeeds.

## Constraints

- Do not add `skills/optim-agent/SKILL.md` or any other duplicate skill file.
- Do not publish under a personal ClawHub namespace.
- Do not store Hub credentials, tokens, temporary archives, or login state in
  the repository.
- Do not change package code or benchmark artifacts as part of distribution.
