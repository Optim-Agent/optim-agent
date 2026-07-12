<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../assets/optim-agent-logo-dark.svg">
    <img alt="optim-agent" src="../assets/optim-agent-logo-light.svg" width="500">
  </picture>
</p>

<h1 align="center">optim-agent</h1>

<p align="center">
  <strong>Agentische Systemoptimierung mit Coding-Agenten.</strong><br>
  Automatisiert die iterative Parameterabstimmung eines Algorithmus-Engineers.
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

optim-agent nutzt Claude Code, Codex oder OpenCode, um jedes System mit
**konfigurierbaren Parametern** und einem **messbaren Ziel** zu optimieren. Das
Werkzeug verbindet die Bedeutung der Parameter mit dem bisherigen
Versuchsverlauf und schlägt die nächste auszuwertende Konfiguration vor. Ihre
Zielfunktion bleibt entscheidend; ungültige Antworten werden geprüft und durch
sicheres Sampling ersetzt.

## Warum optim-agent

- **Semantische Vorschläge**: Der Agent berücksichtigt Parameterbedeutungen,
  Studienkontext und beobachtete Ergebnisse statt nur anonyme Koordinaten.
- **Stark bei kleinen Budgets**: Geeignet, wenn Auswertungen teuer sind und
  klassischen Surrogatmodellen noch Daten fehlen.
- **Nachvollziehbar**: Konfigurationen, Ergebnisse, Zustände, Kontext und
  optionales Agenten-Rationale werden in JSON oder SQLite gespeichert.
- **Begrenzte Ausführung**: Der Agent schlägt nur Werte vor; der Suchraum
  validiert sie und die Zielfunktion entscheidet über das Ergebnis.

## Installation

```bash
pip install optim-agent
```

Zusätzlich muss ein authentifiziertes `claude`-, `codex`- oder `opencode`-CLI im PATH liegen.

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
        context="compute or operating budget",
    )
    return evaluate_system(threshold=threshold, budget=budget)

study = oa.create_study(
    direction="maximize",
    sampler=oa.AgentSampler(
        backend="codex",  # alternativ "claude" / "opencode"
        effort="high",
        context="maximize quality under a strict operating-cost budget",
    ),
    storage="study.json",
)
study.optimize(objective, n_trials=20)
print(study.best_value, study.best_params)
```

`context` ist optional, aber wirkungsvoll. Beschreiben Sie das Gesamtsystem und
die Bedeutung jedes `suggest_*`-Parameters, damit der Agent wie ein
Algorithmus-Engineer statt wie ein blinder Punktsucher schlussfolgern kann.

## Einsatzgebiete

| Bereich | Beispielparameter | Beispielziele |
|---|---|---|
| Modelltraining | Lernraten, Architekturen, Augmentierung, Regularisierung | Qualität, Rechenaufwand, Robustheit |
| Inferenz und Serving | Quantisierung, Batching, Decoding, Caching, Routing | Qualität, Latenz, Durchsatz, Kosten |
| Quantitative Forschung | Signalfenster, Schwellenwerte, Rebalancing, Risikoregeln | Walk-forward-Rendite, Drawdown, Turnover |
| Reinforcement Learning | Zielgewichte, Explorationspläne, Policy-Schwellen | Return, Sicherheit, Sample-Effizienz |
| Wissenschaftliche Abläufe | Simulationseingaben, Solver, Versuchssteuerung | Fit, Fehler, Laufzeit, Ressourcen |
| Black-Box-Systeme | jede begrenzte kategoriale, ganzzahlige oder stetige Konfiguration | jeder messbare Skalarwert |

## Benchmarks

![MNIST- und CIFAR-10-Benchmarks über fünf Seeds](../assets/classification_benchmarks.png)

Random, Optuna TPE sowie Agenten mit und ohne Kontext werden mit identischem
Suchraum, Budget und Seeds verglichen. Exakte Werte, Methodik und
Reproduktionsbefehle stehen im maßgeblichen
[englischen README](../../README.md#benchmarks-agents-vs-tpe-and-random-search).

Weitere Beispiele:

- [Inferenz-Tuning](../../examples/inference_tuning.py)
- [Walk-forward-Tuning quantitativer Signale](../../examples/quant_walk_forward.py)
- [scikit-learn-Tuning](../../examples/sklearn_tuning.py)
- [MNIST](../../examples/mnist.py) / [CIFAR-10](../../examples/cifar10.py)

## Aktuelle Grenzen

- Derzeit wird ein einzelnes Ziel unterstützt. Für mehrere Ziele muss explizit
  eine skalare Nutzenfunktion oder eine Constraint-Strafe definiert werden.
- Bei sehr günstigen Auswertungen mit Tausenden möglichen Versuchen können TPE,
  Gaußprozesse oder evolutionäre Verfahren geeigneter sein.
- Für Reproduzierbarkeit sollten Seeds fixiert und vollständige Studien gespeichert werden.

## Mitwirken

Beiträge sind willkommen. Größere Änderungen bitte vor einem Pull Request in
einem Issue diskutieren. Das [englische README](../../README.md) ist die
maßgebliche Quelle für Versionen, Benchmarkwerte und unterstützte Backends.

## Danksagung

- [Optuna](https://github.com/optuna/optuna) für die Verbreitung der
  Study/Trial-Schnittstelle und die in Beispielen und Benchmarks verwendete TPE-Baseline.

## Lizenz

[MIT](../../LICENSE)
