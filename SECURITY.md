# Security policy

## Supported versions

Security fixes are provided for the latest release on PyPI and the current
`main` branch. Older releases may be asked to upgrade before a fix is issued.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Email
`admin@optim-agent.com` with:

- the affected version or commit;
- a minimal reproduction;
- the impact and required preconditions; and
- any suggested mitigation.

Avoid including API keys, private training data, proprietary prompts, or full
agent transcripts. You should receive an acknowledgement within 5 business
days.

## Response process

We will reproduce the issue, assess severity, coordinate a fix, and agree on a
disclosure timeline. Confirmed high-severity issues take priority over feature
work. Credit is offered unless the reporter prefers anonymity.

## Security model

optim-agent launches authenticated local agent CLIs as subprocesses. Those
tools inherit the permissions and provider configuration of the user running
them. Treat study and parameter context as prompt content: do not include
secrets or untrusted instructions. Agent-proposed values are parsed and
validated against the declared search space, but the user's objective function
remains arbitrary local code and must be reviewed independently.
