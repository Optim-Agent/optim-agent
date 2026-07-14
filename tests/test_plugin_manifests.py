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


def test_readme_documents_claude_and_codex_plugin_installs():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    commands = (
        "claude plugin marketplace add Optim-Agent/optim-agent",
        "claude plugin install optim-agent@optim-agent",
        "codex plugin marketplace add Optim-Agent/optim-agent",
        "codex plugin add optim-agent@optim-agent",
    )
    for command in commands:
        assert command in readme
