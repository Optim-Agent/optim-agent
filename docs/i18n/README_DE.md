<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../assets/optim-agent-logo-dark.svg">
    <img alt="optim-agent" src="../assets/optim-agent-logo-light.svg" width="500">
  </picture>
</p>

<h1 align="center">optim-agent</h1>

<p align="center">
  <strong>Agentische Systemoptimierung mit Coding Agents.</strong><br>
  Automatisiert die iterative Parameterabstimmung eines Algorithmus-Engineers.
</p>

<p align="center">
  <a href="https://github.com/Optim-Agent/optim-agent/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Optim-Agent/optim-agent?style=square"></a>
  <a href="https://pypi.org/project/optim-agent/"><img alt="PyPI" src="https://img.shields.io/pypi/v/optim-agent"></a>
  <a href="https://pypi.org/project/optim-agent/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/optim-agent"></a>
  <a href="../../LICENSE"><img alt="License: MIT" src="https://img.shields.io/pypi/l/optim-agent"></a>
  <a href="https://optim-agent.github.io/optim-agent/"><img alt="Docs" src="https://img.shields.io/badge/docs-online-blue"></a>
  <a href="https://code.claude.com/docs/en/skills"><img alt="Claude Skill" src="https://img.shields.io/badge/Claude-Skill-D97757?logo=claude&logoColor=white"></a>
  <a href="https://developers.openai.com/codex/skills"><img alt="Codex Skill" src="https://img.shields.io/badge/Codex-Skill-blue?logo=openai&logoColor=white"></a>
</p>

<p align="center">
  <a href="../../README.md">English</a> |
  <a href="README_ZH.md">简体中文</a> |
  <a href="README_JA.md">日本語</a> |
  <a href="README_KO.md">한국어</a> |
  <a href="README_FR.md">Français</a> |
  <strong>Deutsch</strong> |
  <a href="README_ES.md">Español</a> |
  <a href="README_PT.md">Português</a> |
  <a href="README_RU.md">Русский</a>
</p>

optim-agent lässt Claude Code / Codex / OpenCode echte Systemparameter abstimmen,
indem es Code liest, Trials vorschlägt und gemessene Objective-Ergebnisse aufzeichnet.
Nutzen Sie es, wenn Ihr System konfigurierbare Parameter und ein messbares Objective bietet.
Es verbindet die Bedeutung jedes Parameters mit den Signalen aus der Trial-Historie und schlägt
die nächste zu bewertende Konfiguration vor. Objective-Auswertungen bleiben maßgeblich:
optim-agent schlägt Werte vor, validiert sie gegen den deklarierten Raum, zeichnet Ergebnisse auf
und fällt bei ungültigen Agent-Antworten auf sicheres Sampling zurück.

<p align="center">
  <img alt="optim-agent tuning loop" src="../assets/optim-agent-overview.png" width="900">
</p>

| Modelle | Systeme | Forschung |
|---|---|---|
| Training, Architekturen und RL-Experimente | Inferenz, Latenz, Kosten, Steuerung und Entscheidungsregeln | Quant-Signale, Simulationen und wissenschaftliche Workflows |

## Warum optim-agent

- **Semantische Vorschläge** - Coding Agents nutzen Parameterbedeutungen, Kontext und beobachtete Ergebnisse, statt jede Dimension als anonyme Koordinate zu behandeln.
- **Hebel bei kleinem Budget** - hilfreich, wenn Evaluierungen teuer sind und klassische Surrogate noch zu wenig Daten haben.
- **Agent-CLI-Potenzial** - die Vorschlagsqualität kann mit besseren Coding Agents steigen, etwa von GPT-5.5 zu GPT-5.6, ohne Optimierungscode zu ändern.
- **Auditierbare Entscheidungen** - JSON/SQLite-Studies behalten Konfigurationen, Ergebnisse, Zustände, Kontext und optionale Agent-Begründungen.
- **Begrenzte Ausführung** - der Agent schlägt nur Werte vor; optim-agent validiert sie gegen den Suchraum, ungültige Ausgaben fallen auf sicheres Sampling zurück.

## Installation

Codex Skill installieren:

```text
$skill-installer install https://github.com/Optim-Agent/optim-agent
```

Claude Code Plugin installieren:

```bash
claude plugin marketplace add Optim-Agent/optim-agent && claude plugin install optim-agent@optim-agent
```

Python-Paket installieren:

```bash
# Stabile Version von PyPI
python -m pip install optim-agent

# Neuester Quellstand von GitHub
python -m pip install "optim-agent @ git+https://github.com/Optim-Agent/optim-agent.git"
```

Erfordert mindestens eine authentifizierte Agent CLI in `PATH`:
[claude](https://docs.anthropic.com/en/docs/claude-code),
[codex](https://github.com/openai/codex) oder
[OpenCode](https://github.com/sst/opencode).

## Schnellstart

```python
import optim_agent as oa

def objective(trial):
    threshold = trial.suggest_float(
        "threshold", 0.05, 0.95,
        context="decision threshold; higher values trade recall for precision",
    )
    budget = trial.suggest_int(
        "budget", 10, 200, log=True,
        context="compute or operating budget; larger values may improve quality",
    )
    return evaluate_system(threshold=threshold, budget=budget)  # domain code

study = oa.create_study(
    direction="maximize",
    sampler=oa.AgentSampler(
        backend="claude",  # or "codex" / "opencode"
        effort="high",
        context="maximize system quality under a strict operating-cost budget",
        history=5,
        explicit_reasoning=True,
        qualitative_notes=True,
    ),
    storage="study.json",  # optional: persist and resume
)
study.optimize(objective, n_trials=20)
print(study.best_value, study.best_params)
```

Optionaler `context` gibt Study und Parametern fachliche Bedeutung. Setzen Sie ihn auf Study-Ebene mit
`AgentSampler(context=...)`, pro Parameter mit `suggest_*(..., context=...)` oder beides.

Sie können auch [`examples/quickstart.py`](../../examples/quickstart.py) ausführen oder
[`tutorials/quickstart.ipynb`](../../tutorials/quickstart.ipynb) nutzen.

## Einsatzgebiete

| Bereich | Parameter, die optim-agent abstimmen kann | Beispiel-Objective |
|---|---|---|
| **Modelltraining** | Lernraten, Architekturen, Augmentation, Regularisierung | Validierungsqualität, Rechenaufwand, Robustheit |
| **Inferenz und Serving** | Quantisierung, Batching, Decoding, Caching, Routing | Qualität, Latenz, Durchsatz, Kosten |
| **Quantitative Forschung** | Signal-Fenster, Schwellen, Rebalancing-Regeln, Risikokontrollen | Walk-forward Return, Drawdown, Turnover |
| **RL und Entscheidungen** | Objective-Gewichte, Explorationspläne, Umgebungssettings, Policy-Schwellen | Return, Sicherheit, Sample Efficiency |
| **Wissenschaftliche Workflows** | Simulationseingaben, Solver-Settings, Experimentkontrollen | Fit, Fehler, Laufzeit, Ressourcen |
| **Black-box-Systeme** | jede begrenzte kategoriale, ganzzahlige oder kontinuierliche Konfiguration | skalarer Objective-Score |

Weitere Beispiele: [`examples/sklearn_tuning.py`](../../examples/sklearn_tuning.py) und
[`examples/inference_tuning.py`](../../examples/inference_tuning.py).

Bei Reinforcement Learning stimmt optim-agent das System um den Lernloop herum ab; es ersetzt nicht den Policy-Learning-Algorithmus.

## Optimierungstrajektorie

![Agent optimization trajectory compared with TPE](../assets/optimization_trajectory.gif)

Diese Seed-0-Branin-Trajektorie vergleicht TPE und GPT-5.5 mit demselben 10-Trial-Budget
und zeigt die incumbent Objective-Werte nach jedem Trial. Sie ist eine Trajektorien-Illustration;
aggregierte Benchmark-Ergebnisse und Reproduktionsbefehle folgen.

### Mathematische Funktionen ohne Kontext optimieren: Branin-2D und Ackley-5D

Hard-function Agents erhalten **keinen bereitgestellten Task-Kontext**: nur generische
Parameternamen `x1...x5`, numerische Grenzen und Trial-Historie. Runs verwenden 10 Trials
über fünf Seeds; Random und TPE sind unveränderte Baselines.

#### Top-tier Agents

![No-context top-tier hard-function benchmark](../assets/hard_benchmarks_tier.png)

| Methode | mean best Branin ↓ | mean best Ackley-5D ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| GPT-5.5 | 1.326 | 3.960 |
| **Opus-4.8** | **0.398** | **0.061** |
| Sonnet-5 | 3.850 | 0.143 |
| GLM-5.2 | 3.609 | 15.023 |

Die fixierten Modelle sind `gpt-5.5`, `claude-opus-4-8`, `claude-sonnet-5` und `glm-5.2`.
Opus-4.8 erreicht den Branin-Optimumbereich im Mittel und hat den stärksten fünf-Seed-Ackley-Mittelwert.

#### OpenCode Agents (Free)

![No-context free-model hard-function benchmark](../assets/hard_benchmarks_free.png)

| Methode | mean best Branin ↓ | mean best Ackley-5D ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| Big-pickle | 4.734 | 15.951 |
| **DeepSeek-V4-Flash** | 4.410 | **4.608** |
| Nemotron-3-Ultra | 16.051 | 18.459 |
| **MiMo-v2.5** | **3.682** | 15.597 |

OpenCode-gehostete Modelle benötigen keine kostenpflichtige Modell-API. Der kostenlose Pool rotiert;
dieser Refresh pinnt `opencode/big-pickle`, `opencode/deepseek-v4-flash-free`,
`opencode/nemotron-3-ultra-free` und `opencode/mimo-v2.5-free`. DeepSeek V4 Flash hat den stärksten
Free-model-Ackley-Mittelwert, MiMo-v2.5 den stärksten Free-model-Branin-Mittelwert.

### ResNet-Bildklassifikatoren tunen: MNIST und CIFAR-10

Der Klassifikationsbenchmark vergleicht **Random**, Optuna **TPE**, **GPT-5.5 w/ context**
und **GPT-5.5 w/o context** über fünf Seeds (`0..4`) und 10 Trials. Die Kontextbedingung erhält
natürlichsprachliche Study- und Parameterbeschreibungen; die No-context-Bedingung nur Grenzen und Trial-Historie.

Die Hauptmetrik betont schnelle Verbesserung:

```text
cumulative_best_so_far_error = sum(best_test_error_so_far_at_i for i in 1..10)
```

Niedriger ist besser.

![MNIST- und CIFAR-10-Benchmarks über fünf Seeds](../assets/classification_benchmarks.png)

| Methode | MNIST cumulative error ↓ | MNIST final error ↓ | CIFAR-10 cumulative error ↓ | CIFAR-10 final error ↓ |
|---|---:|---:|---:|---:|
| Random | 9.174 | 0.648% | 278.920 | 25.072% |
| TPE | 7.166 | 0.580% | 279.936 | 25.596% |
| **GPT-5.5 w/ context** | **5.668** | **0.506%** | **220.994** | **21.322%** |
| GPT-5.5 w/o context | 8.910 | 0.632% | 281.466 | 25.960% |

GPT-5.5 w/ context senkt den cumulative best-so-far error um **20.9%** gegenüber TPE auf MNIST
und um **20.8%** gegenüber Random auf CIFAR-10. Ohne Kontext ist es auf MNIST 24.3% schlechter als TPE
und auf CIFAR-10 0.9% schlechter als Random.

[`examples/mnist.py`](../../examples/mnist.py) und [`examples/cifar10.py`](../../examples/cifar10.py)
tunen Lernrate, Batchgröße, Weight Decay, Label Smoothing, drei Stage-Breiten, drei Stage-Tiefen
und vier Dropout-Kontrollen. MNIST ergänzt Translation und Rotation; CIFAR-10 nutzt Crop Padding und Flip Probability.

### Q-learning Controller tunen: Acrobot-v1 und LunarLander-v3

![CPU-only Gymnasium RL control benchmark](../assets/rl_control.png)

Dieser CPU-only Gymnasium Benchmark tuned einen diskretisierten Q-learning Controller für Acrobot-v1 und LunarLander-v3.
Jede Methode läuft 20 Trials über fünf Seeds (`0..4`); das Objective ist der mittlere Evaluierungs-Return,
also ist höher besser. Der Runner parallelisiert über Seeds und innerhalb jeder HPO-Study via `--workers`.
Die GPT-5.5-Arme nutzen high modeling effort und die letzten 5 Trials Historie.

| Methode | Acrobot-v1 Return ↑ | LunarLander-v3 Return ↑ |
|---|---:|---:|
| Random | -200.000 | -62.139 |
| TPE | -199.900 | -72.088 |
| **GPT-5.5 w/ context** | **-199.700** | **-50.825** |
| GPT-5.5 w/o context | -199.100 | -59.751 |

Mit 20 Trials und fünf Trial-Historie im Prompt hat GPT-5.5 w/ context den stärksten mittleren Return
in beiden Umgebungen: 0.2 über TPE auf Acrobot-v1 und 11.3 über Random auf LunarLander-v3.
Behandeln Sie dies als CPU-HPO-Stresstest, nicht als universelles Ranking.

Für die Animation tuned optim-agent sieben Gains eines deterministischen LunarLander Controllers mit einem HPO-Seed.
Jeder Trial läuft auf denselben 20 Rollout-Seeds, priorisiert erfolgreiche Landungen und danach mittleren Return.
Der ausgewählte Trial landete in allen 20 Rollouts; das GIF zeigt seinen Rollout mit dem höchsten Return.

![LunarLander rollout from a committed GPT-5.5 policy](../assets/lunarlander_policy.gif)

### Gradient-Boosting-Klassifikator tunen: Kreditausfallwahrscheinlichkeiten

![Five-seed CPU-only GPT-5.5 context benchmark for UCI credit-default HGB tuning](../assets/credit_card.png)

Dieser CPU-only Benchmark tuned acht Trainingsparameter eines `HistGradientBoostingClassifier` auf dem UCI-Datensatz
[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients):
30.000 Zeilen, 23 Merkmale und Ziel "Default im nächsten Monat". Das offizielle Archiv ist per SHA-256 gepinnt,
CC BY 4.0 lizenziert und einmal in 60% Train, 20% Validation und 20% untouched Test aufgeteilt.
Alle Methoden nutzen dieselbe Aufteilung, 20 Trials und Seeds `0..4`. Beide GPT-5.5-Arme verwenden
high modeling effort, 20 Trials Prompt-Historie, explicit reasoning und qualitative notes.

| Methode | finaler Validierungs-Log-Loss ↓ | held-out Test-Log-Loss ↓ |
|---|---:|---:|
| Random | 0.433 | 0.425 |
| TPE | 0.430 | 0.422 |
| GP-BO | 0.430 | 0.423 |
| **GPT-5.5 w/ context** | **0.428** | **0.422** |
| GPT-5.5 w/o context | 0.433 | 0.427 |

Kontext senkt den finalen Validierungs-Log-Loss um 1,13% und den Test-Log-Loss um 1,23%
gegenüber der passenden No-context-Kontrolle. GPT-5.5 hat auch niedrigeren mittleren Validierungs-
und Test-Loss als Random, TPE und GP-BO. Da die behaltene Konfiguration mit Validation und Test Loss gewählt wurde,
ist das Testergebnis ein Benchmarkvergleich und keine unberührte Generalisierungsschätzung.

Dies ist ein methodischer Benchmark, kein produktives Kreditentscheidungssystem. Deployment bräuchte Fairness,
Kalibrierung, Drift, Governance und juristische Prüfung.

Benchmark-Artefakte reproduzieren:

```bash
pip install -e ".[examples]"

# Classification
python scripts/verify_classification_cumulative_error.py run-no-context
python scripts/verify_classification_cumulative_error.py

# Hard functions
python examples/hard_functions.py preflight
python examples/hard_functions.py distributed --trials 10 --seeds 0 1 2 3 4
python examples/hard_functions.py plot

# Credit-card HGB
pip install -e ".[ml,examples]"
python examples/credit_card.py download
python examples/credit_card.py preflight
python examples/credit_card.py run
python examples/credit_card.py selfcheck
python examples/credit_card.py summary
python examples/credit_card.py plot

# RL control
pip install -e ".[rl,examples]"
python examples/rl_control.py preflight
python examples/rl_control.py run --seeds 0 1 2 3 4 --workers 10
python examples/rl_control.py selfcheck
python examples/rl_control.py summary
python examples/rl_control.py plot
python examples/rl_control.py gif
```

## Nutzungsleitfaden

### Sampler Prompt Controls

`effort` wird an das reasoning-effort-Flag der Backend-CLI weitergereicht. Der Harness-Prompt wird separat gesteuert:

```python
oa.AgentSampler(
    backend="codex",
    effort="medium",
    history=5,
    explicit_reasoning=True,
    qualitative_notes=True,
)
```

Setzen Sie `history=None`, um alle abgeschlossenen/geprunten Trials zu zeigen. Nutzen Sie
`explicit_reasoning=False` oder `qualitative_notes=False` für kürzere Agent-Antworten.

### Pruning

```python
study = oa.create_study(
    sampler=oa.AgentSampler(backend="codex"),
    pruner=oa.AgentPruner(
        backend="codex", level="medium", effort="medium",
    ),  # level: loose | medium | tight
)

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True,
                             context="learning rate for training an image classifier")
    for epoch in range(20):
        loss = train_one_epoch(lr)
        trial.report(loss, epoch)
        if trial.should_prune():
            raise oa.TrialPruned()
    return loss
```

Der Pruner-Agent vergleicht die aktuelle Lernkurve mit abgeschlossenen Trials und antwortet prune/keep;
`loose` pruned nur klar schwache Runs, `tight` pruned aggressiver. Agent-Fehler prunen nie einen Trial.

### Nebenläufige und verteilte Studies

Setzen Sie `max_concurrency` (Standard `1`), um mehrere Trials gleichzeitig zu evaluieren, und nutzen Sie
eine SQLite-`storage`-Datei (`.db` / `.sqlite`) als nebenläufigkeitssichere gemeinsame Historie:

```python
study = oa.create_study(
    sampler=oa.AgentSampler(backend="claude"),
    storage="study.db",        # SQLite -> safe for many workers; .json stays single-writer
    max_concurrency=8,         # up to 8 objectives run at once
)
study.optimize(objective, n_trials=100)
```

- **Innerhalb eines Prozesses** führt `max_concurrency` Objectives in einem Threadpool aus. Agent-Sampling-Queries
  werden seriell in eine Queue gelegt, damit jeder Vorschlag die Prozesshistorie sieht; nur Objective-Aufrufe laufen parallel.
- **Über Prozesse/Maschinen hinweg** zeigen alle Worker auf dieselbe SQLite-`storage`. Die Datenbank ist der Kommunikationskanal:
  WAL-Modus lässt Worker Ergebnisse anhängen und Historie lesen, ohne Schreibkonflikte.

Einschränkungen: Threads teilen den GIL, daher laufen pure-Python CPU-bound Objectives am besten in getrennten Prozessen mit geteilter SQLite Storage.
Nebenläufige Worker sehen die in-flight Punkte der anderen nicht und können gelegentlich nahe Regionen prüfen.

### Skill-Modus (Agent liest Projektcode)

Das pip-Paket behandelt das Objective als Black Box. Der [optim-agent Skill](../../SKILL.md) geht weiter:
In einer Coding-Agent-Session liest der Agent zuerst das Projekt, versteht die Rolle jedes Parameters und steuert
denselben Study-Loop über `study.ask(params)` / `study.tell(trial, value)`, wobei das Study-JSON Historie über Sessions hält.

```text
$skill-installer install https://github.com/Optim-Agent/optim-agent
```

Claude Code Plugin:

```bash
claude plugin marketplace add Optim-Agent/optim-agent
claude plugin install optim-agent@optim-agent
```

Codex Plugin:

```bash
codex plugin marketplace add Optim-Agent/optim-agent
codex plugin add optim-agent@optim-agent
```

```python
trial = study.ask({"threshold": 0.72, "budget": 80})
study.tell(trial, evaluate_system(**trial.params))
```

### Offline Testing

`AgentSampler(backend="mock")` ist ein tokenfreier Ersatz, der um den besten Punkt hill-climbt,
um Integrationen vor echten Agent Calls zu testen.

## Fehlerbehebung

- **`claude` gibt 401 in einer Agent-Session zurück** - verschachtelte Sessions erben `ANTHROPIC_API_KEY`;
  starten Sie mit `env -u ANTHROPIC_API_KEY` oder aus einer sauberen Shell.
- **Ein Backend Call läuft in ein Timeout oder liefert ungültige Ausgabe** - der Sampler warnt und fällt für diesen Trial auf einen Random Point zurück; die Study läuft weiter.
- **OpenCode mit verteilten Studies** - OpenCode currently does not support distributed computing
  in optim-agent; nutzen Sie den Single-Process-Workflow oder ein anderes Backend.

## Mitwirken

Lokale Entwicklung:

```bash
pip install -e ".[examples]"
pytest                     # runs tests/test_optim_agent.py
```

Bitte größere Änderungen vor einem PR in einem Issue besprechen. Einen neuen Agent Backend hinzuzufügen heißt meist:
eine kleine Funktion in [`optim_agent/agent.py`](../../optim_agent/agent.py).

Die englische [`README.md`](../../README.md) bleibt die maßgebliche Quelle für Versionen, Benchmarkwerte und Backends.

## Danksagung

- [Optuna](https://github.com/optuna/optuna) für die Verbreitung der Study/Trial-Schnittstelle, die TPE-Baseline in Beispielen
  und Benchmarks und den hohen Standard für praktische Optimierungstools.
- [OpenCode](https://github.com/sst/opencode) für den Zugang zu den kostenlosen Modellen in den Hard-function Benchmarks.

## Lizenz

[MIT](../../LICENSE)
