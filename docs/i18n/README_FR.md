<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../assets/optim-agent-logo-dark.svg">
    <img alt="optim-agent" src="../assets/optim-agent-logo-light.svg" width="500">
  </picture>
</p>

<h1 align="center">optim-agent</h1>

<p align="center">
  <strong>Optimisation agentique de systèmes avec des agents de programmation.</strong><br>
  Automatisez le réglage itératif des paramètres effectué par un ingénieur algorithmique.
</p>

<p align="center">
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
  <strong>Français</strong> |
  <a href="README_DE.md">Deutsch</a> |
  <a href="README_ES.md">Español</a> |
  <a href="README_PT.md">Português</a> |
  <a href="README_RU.md">Русский</a>
</p>

optim-agent permet à Claude Code / Codex / OpenCode d'ajuster des paramètres
système réels en lisant votre code, en proposant des essais et en enregistrant
les résultats objectifs mesurés. Utilisez-le lorsque votre système expose des
paramètres configurables et un objectif mesurable. Il combine le sens de chaque
paramètre avec ce que montre l'historique des essais, puis propose la prochaine
configuration à évaluer. Les évaluations de l'objectif restent souveraines :
optim-agent propose des valeurs, les valide par rapport à l'espace déclaré,
enregistre les résultats et revient à un échantillonnage sûr lorsqu'une réponse
d'agent est invalide.

## Pourquoi optim-agent

- **Propositions sémantiques** : l'agent raisonne sur le sens des paramètres,
  le contexte de l'étude et les résultats observés.
- **Efficace avec un petit budget** : utile lorsque chaque évaluation est
  coûteuse et que les modèles de substitution classiques manquent de données.
- **Traçable** : configurations, résultats, états, contexte et raisonnement
  facultatif sont conservés en JSON ou SQLite.
- **Exécution bornée** : l'agent propose seulement des valeurs ; l'espace de
  recherche les valide et la fonction objectif décide du résultat.

## Installation

Installez le skill Codex :

```text
$skill-installer install https://github.com/Optim-Agent/optim-agent
```

Installez le plugin Claude Code :

```bash
claude plugin marketplace add Optim-Agent/optim-agent && claude plugin install optim-agent@optim-agent
```

Installez le package Python :

```bash
# Version stable sur PyPI
python -m pip install optim-agent

# Dernière version source sur GitHub
python -m pip install "optim-agent @ git+https://github.com/Optim-Agent/optim-agent.git"
```

Ajoutez aussi à votre PATH un CLI `claude`, `codex` ou `opencode` déjà authentifié.

## Démarrage rapide

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
        backend="codex",  # ou "claude" / "opencode"
        effort="high",
        context="maximize quality under a strict operating-cost budget",
        history=5,
        explicit_reasoning=True,
        qualitative_notes=True,
    ),
    storage="study.json",
)
study.optimize(objective, n_trials=20)
print(study.best_value, study.best_params)
```

`context` est facultatif mais puissant. Décrivez le système global et chaque
paramètre `suggest_*` pour permettre à l'agent de raisonner comme un ingénieur
algorithmique plutôt que comme un simple explorateur de points.

## Mode skill

Le mode package traite l'objectif comme une boîte noire. Avec le
[`SKILL.md`](../../SKILL.md) à la racine, l'agent de programmation actif lit
d'abord le projet et comprend les relations entre paramètres, puis pilote la
même étude avec `study.ask(params)` et `study.tell(trial, value)`. Chargez le
skill directement depuis GitHub dans l'environnement actif de l'agent de programmation :

```text
$skill-installer install https://github.com/Optim-Agent/optim-agent
```

## Domaines d'application

| Domaine | Exemples de paramètres | Exemples d'objectifs |
|---|---|---|
| Entraînement de modèles | taux d'apprentissage, architectures, augmentation, régularisation | qualité, calcul, robustesse |
| Inférence et service | quantification, lots, décodage, cache, routage | qualité, latence, débit, coût |
| Recherche quantitative | fenêtres de signal, seuils, rééquilibrage, contrôle du risque | rendement walk-forward, drawdown, rotation |
| Apprentissage par renforcement | poids d'objectifs, exploration, seuils de politique | retour, sécurité, efficacité d'échantillonnage |
| Recherche scientifique | entrées de simulation, solveurs, contrôles expérimentaux | ajustement, erreur, temps, ressources |
| Systèmes boîte noire | toute configuration bornée catégorielle, entière ou continue | toute valeur scalaire mesurable |

## Benchmarks

### Optimisation de fonctions mathématiques sans contexte : Branin-2D et Ackley-5D

Les agents de fonctions difficiles ne reçoivent **aucun contexte de tâche fourni** : seulement des noms génériques `x1...x5`, des bornes numériques et l'historique des essais. Tous utilisent un effort medium, 10 essais et cinq graines ; Random et TPE restent les lignes de base.

#### Agents haut de gamme

![No-context top-tier hard-function benchmark](../assets/hard_benchmarks_tier.png)

| méthode | meilleur Branin moyen ↓ | meilleur Ackley-5D moyen ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| GPT-5.5 | 1.326 | 3.960 |
| **Opus-4.8** | **0.398** | **0.061** |
| Sonnet-5 | 3.850 | 0.143 |
| GLM-5.2 | 3.609 | 15.023 |

#### Agents OpenCode (gratuits)

![No-context free-model hard-function benchmark](../assets/hard_benchmarks_free.png)

| méthode | meilleur Branin moyen ↓ | meilleur Ackley-5D moyen ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| Big-pickle | 4.734 | 15.951 |
| DeepSeek-V4-Flash | 4.410 | **4.608** |
| Nemotron-3-Ultra | 16.051 | 18.459 |
| **MiMo-v2.5** | **3.682** | 15.597 |

### Réglage de classificateurs d'images ResNet : MNIST et CIFAR-10

![Benchmarks MNIST et CIFAR-10 sur cinq graines](../assets/classification_benchmarks.png)

Random, Optuna TPE, **GPT-5.5 w/ context** et **GPT-5.5 w/o context** sont comparés sur cinq graines (`0..4`) et 10 essais. Les deux conditions GPT-5.5 fixent `gpt-5.5` avec un effort de raisonnement medium (`model_reasoning_effort=medium`) ; seule **GPT-5.5 w/ context** reçoit les descriptions en langage naturel de l'objectif et des 16 paramètres.

| méthode | erreur cumulative MNIST ↓ | erreur finale MNIST ↓ | erreur cumulative CIFAR-10 ↓ | erreur finale CIFAR-10 ↓ |
|---|---:|---:|---:|---:|
| Random | 9.174 | 0.648% | 278.920 | 25.072% |
| TPE | 7.166 | 0.580% | 279.936 | 25.596% |
| **GPT-5.5 w/ context** | **5.668** | **0.506%** | **220.994** | **21.322%** |
| GPT-5.5 w/o context | 8.910 | 0.632% | 281.466 | 25.960% |

### Réglage d'un classificateur gradient boosting : probabilités de défaut de crédit

![Five-seed CPU-only GPT-5.5 context benchmark for UCI credit-default HGB tuning](../assets/credit_card.png)

Ce benchmark CPU-only règle huit paramètres d'entraînement d'un `HistGradientBoostingClassifier` sur UCI **Default of Credit Card Clients** (30 000 lignes, 23 variables, CC BY 4.0). Toutes les méthodes utilisent la même partition, 20 essais et les graines `0..4`.

| méthode | log loss final de validation ↓ | log loss test holdout ↓ |
|---|---:|---:|
| Random | 0.433 | 0.425 |
| TPE | 0.430 | **0.422** |
| **GPT-5.5 w/ context** | **0.428** | **0.422** |
| GPT-5.5 w/o context | 0.433 | 0.427 |

Avec la configuration GPT-5.5 sélectionnée, le contexte réduit le log loss final de validation de 1,16 % et le log loss de test holdout de 1,15 % face au contrôle sans contexte. C'est un benchmark méthodologique, pas un système de décision de crédit en production.

Autres exemples :

- [Réglage de l'inférence](../../examples/inference_tuning.py)
- [Réglage scikit-learn](../../examples/sklearn_tuning.py)
- [MNIST](../../examples/mnist.py) / [CIFAR-10](../../examples/cifar10.py)

## Limites actuelles

- L'optimisation est actuellement mono-objectif. Pour plusieurs objectifs,
  définissez explicitement une utilité scalaire ou des pénalités de contrainte.
- Pour des évaluations très bon marché autorisant des milliers d'essais, TPE,
  les processus gaussiens ou les méthodes évolutionnaires peuvent être préférables.
- Pour la reproductibilité, fixez les graines et conservez l'étude complète.

## Dépannage

- **OpenCode et études distribuées** : OpenCode ne prend actuellement pas en
  charge le workflow `distributed computing` d'optim-agent. Utilisez le mode
  mono-processus ou un autre backend pour les exécutions distribuées.

## Contribution

Les contributions sont bienvenues. Discutez des changements importants dans
une issue avant d'ouvrir une Pull Request. Le [README anglais](../../README.md)
est la référence pour les versions, les résultats et les backends pris en charge.

## Remerciements

- [Optuna](https://github.com/optuna/optuna), qui a popularisé l'interface
  Study/Trial et fournit la référence TPE utilisée dans les exemples et benchmarks.
- [OpenCode](https://github.com/sst/opencode), qui donne accès aux modèles
  gratuits évalués dans les benchmarks de fonctions difficiles.

## Licence

[MIT](../../LICENSE)
