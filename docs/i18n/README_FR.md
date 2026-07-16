<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../assets/optim-agent-logo-dark.svg">
    <img alt="optim-agent" src="../assets/optim-agent-logo-light.svg" width="500">
  </picture>
</p>

<h1 align="center">optim-agent</h1>

<p align="center">
  <strong>Optimisation agentique de systèmes avec des agents de code.</strong><br>
  Automatise le travail itératif de réglage de paramètres d'un ingénieur algorithme.
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
  <strong>Français</strong> |
  <a href="README_DE.md">Deutsch</a> |
  <a href="README_ES.md">Español</a> |
  <a href="README_PT.md">Português</a> |
  <a href="README_RU.md">Русский</a>
</p>

optim-agent permet à Claude Code / Codex / OpenCode de régler de vrais paramètres système
en lisant votre code, en proposant des trials et en enregistrant les résultats objectifs mesurés.
Utilisez-le quand votre système expose des paramètres configurables et un objectif mesurable.
Il combine ce que chaque paramètre *signifie* avec ce que l'historique des trials *montre*,
puis propose la configuration suivante à évaluer. Les évaluations objectives restent l'autorité :
optim-agent propose des valeurs, les valide contre l'espace déclaré, enregistre les résultats
et revient à un échantillonnage sûr quand une réponse d'agent est invalide.

<p align="center">
  <img alt="optim-agent tuning loop" src="../assets/optim-agent-overview.png" width="900">
</p>

| Modèles | Systèmes | Recherche |
|---|---|---|
| Entraînement, architectures et expériences RL | Inférence, latence, coût, contrôle et règles de décision | Signaux quantitatifs, simulations et workflows scientifiques |

## Pourquoi optim-agent

- **Propositions sémantiques** - les agents de code raisonnent sur le sens des paramètres, le contexte et les résultats observés au lieu de traiter chaque dimension comme une coordonnée anonyme.
- **Effet de levier avec petit budget** - utile quand les évaluations sont coûteuses et que les surrogates classiques manquent encore de données.
- **Potentiel des Agent CLI** - la qualité des propositions peut progresser avec les agents de code sous-jacents, par exemple de GPT-5.5 à GPT-5.6, sans changer le code d'optimisation.
- **Décisions auditables** - les studies JSON/SQLite conservent configurations, résultats, états, contexte et justification optionnelle de l'agent.
- **Exécution bornée** - l'agent ne propose que des valeurs ; optim-agent les valide contre l'espace déclaré et revient à un échantillonnage sûr si la sortie est invalide.

## Installation

Installer le Codex skill :

```text
$skill-installer install https://github.com/Optim-Agent/optim-agent
```

Installer le plugin Claude Code :

```bash
claude plugin marketplace add Optim-Agent/optim-agent && claude plugin install optim-agent@optim-agent
```

Installer le package Python :

```bash
# Version stable sur PyPI
python -m pip install optim-agent

# Dernière version source sur GitHub
python -m pip install "optim-agent @ git+https://github.com/Optim-Agent/optim-agent.git"
```

Nécessite un agent CLI authentifié dans `PATH` :
[claude](https://docs.anthropic.com/en/docs/claude-code),
[codex](https://github.com/openai/codex) ou
[OpenCode](https://github.com/sst/opencode).

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

Le `context` optionnel donne un sens métier à la study et aux paramètres. Fournissez-le
au niveau study avec `AgentSampler(context=...)`, au niveau paramètre avec
`suggest_*(..., context=...)`, ou les deux.

Vous pouvez aussi lancer [`examples/quickstart.py`](../../examples/quickstart.py) ou suivre
[`tutorials/quickstart.ipynb`](../../tutorials/quickstart.ipynb).

## Domaines d'application

| Domaine | Paramètres que optim-agent peut régler | Objectif exemple |
|---|---|---|
| **Entraînement de modèles** | taux d'apprentissage, architectures, augmentation, régularisation | qualité de validation, calcul, robustesse |
| **Inférence et serving** | quantification, batching, décodage, cache, routage | qualité, latence, débit, coût |
| **Recherche quantitative** | fenêtres de signal, seuils, règles de rééquilibrage, contrôles de risque | rendement walk-forward, drawdown, turnover |
| **RL et décisions** | poids d'objectif, calendriers d'exploration, paramètres d'environnement, seuils de politique | retour, sûreté, efficacité échantillon |
| **Workflows scientifiques** | entrées de simulation, paramètres solveur, contrôles expérimentaux | ajustement, erreur, temps, ressources |
| **Systèmes boîte noire** | toute configuration catégorielle, entière ou continue bornée | score objectif scalaire |

Voir aussi [`examples/sklearn_tuning.py`](../../examples/sklearn_tuning.py) et
[`examples/inference_tuning.py`](../../examples/inference_tuning.py).

Pour le reinforcement learning, optim-agent règle le système autour de la boucle d'apprentissage ;
il ne remplace pas l'algorithme d'apprentissage de politique.

## Trajectoire d'optimisation

![Agent optimization trajectory compared with TPE](../assets/optimization_trajectory.gif)

Cette trace Branin seed-0 compare TPE et GPT-5.5 avec le même budget de 10 trials,
en montrant la valeur objective incumbent après chaque trial. C'est une illustration de trajectoire ;
les résultats agrégés et les commandes de reproduction suivent.

### Optimisation de fonctions mathématiques sans contexte : Branin-2D et Ackley-5D

Les agents de fonctions difficiles ne reçoivent **aucun contexte de tâche fourni** : seulement les noms
génériques `x1...x5`, les bornes numériques et l'historique des trials. Les runs utilisent 10 trials
sur cinq seeds ; Random et TPE sont des baselines inchangées.

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

Les modèles épinglés sont `gpt-5.5`, `claude-opus-4-8`, `claude-sonnet-5` et `glm-5.2`.
Opus-4.8 atteint l'optimum Branin en moyenne et obtient la meilleure moyenne Ackley sur cinq seeds.

#### Agents OpenCode (gratuits)

![No-context free-model hard-function benchmark](../assets/hard_benchmarks_free.png)

| méthode | meilleur Branin moyen ↓ | meilleur Ackley-5D moyen ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| Big-pickle | 4.734 | 15.951 |
| **DeepSeek-V4-Flash** | 4.410 | **4.608** |
| Nemotron-3-Ultra | 16.051 | 18.459 |
| **MiMo-v2.5** | **3.682** | 15.597 |

Les modèles hébergés par OpenCode ne demandent pas d'API de modèle payante. Le pool gratuit tourne ;
cette mise à jour épingle `opencode/big-pickle`, `opencode/deepseek-v4-flash-free`,
`opencode/nemotron-3-ultra-free` et `opencode/mimo-v2.5-free`. DeepSeek V4 Flash a la meilleure
moyenne Ackley gratuite, tandis que MiMo-v2.5 a la meilleure moyenne Branin gratuite.

### Réglage de classificateurs d'images ResNet : MNIST et CIFAR-10

Le benchmark de classification compare **Random**, Optuna **TPE**, **GPT-5.5 w/ context**
et **GPT-5.5 w/o context** sur cinq seeds (`0..4`) et 10 trials. La condition avec contexte reçoit
des descriptions naturelles de la study et des paramètres ; la condition sans contexte ne reçoit que les bornes et l'historique.

La métrique principale favorise l'amélioration rapide :

```text
cumulative_best_so_far_error = sum(best_test_error_so_far_at_i for i in 1..10)
```

Plus bas est meilleur.

![Benchmarks MNIST et CIFAR-10 sur cinq graines](../assets/classification_benchmarks.png)

| méthode | erreur cumulative MNIST ↓ | erreur finale MNIST ↓ | erreur cumulative CIFAR-10 ↓ | erreur finale CIFAR-10 ↓ |
|---|---:|---:|---:|---:|
| Random | 9.174 | 0.648% | 278.920 | 25.072% |
| TPE | 7.166 | 0.580% | 279.936 | 25.596% |
| **GPT-5.5 w/ context** | **5.668** | **0.506%** | **220.994** | **21.322%** |
| GPT-5.5 w/o context | 8.910 | 0.632% | 281.466 | 25.960% |

GPT-5.5 w/ context réduit l'erreur cumulative best-so-far de **20.9%** par rapport à TPE sur MNIST
et de **20.8%** par rapport à Random sur CIFAR-10. Sans contexte, il est 24.3% pire que TPE sur MNIST
et 0.9% pire que Random sur CIFAR-10.

[`examples/mnist.py`](../../examples/mnist.py) et [`examples/cifar10.py`](../../examples/cifar10.py)
règlent le learning rate, la batch size, le weight decay, le label smoothing, trois largeurs de stage,
trois profondeurs de stage et quatre contrôles de dropout. MNIST ajoute translation et rotation ;
CIFAR-10 utilise crop padding et flip probability.

### Réglage de contrôleurs Q-learning : Acrobot-v1 et LunarLander-v3

![CPU-only Gymnasium RL control benchmark](../assets/rl_control.png)

Ce benchmark CPU-only Gymnasium règle un contrôleur Q-learning discrétisé pour Acrobot-v1 et LunarLander-v3.
Chaque méthode exécute 20 trials sur cinq seeds (`0..4`) ; l'objectif est le retour d'évaluation moyen,
donc plus haut est meilleur. Le runner parallélise entre seeds et dans chaque study HPO via `--workers`.
Les bras GPT-5.5 utilisent high modeling effort et les 5 derniers trials d'historique.

| méthode | retour Acrobot-v1 ↑ | retour LunarLander-v3 ↑ |
|---|---:|---:|
| Random | -200.000 | -62.139 |
| TPE | -199.900 | -72.088 |
| **GPT-5.5 w/ context** | **-199.700** | **-50.825** |
| GPT-5.5 w/o context | -199.100 | -59.751 |

Avec 20 trials et cinq trials d'historique de prompt, GPT-5.5 w/ context a le meilleur retour moyen
sur les deux environnements : 0.2 au-dessus de TPE sur Acrobot-v1 et 11.3 au-dessus de Random sur LunarLander-v3.
Considérez cela comme un stress test CPU HPO, pas comme un classement universel.

Pour l'animation, optim-agent règle sept gains d'un contrôleur LunarLander déterministe avec un seed HPO.
Chaque trial utilise les mêmes 20 seeds de rollout, en priorisant le nombre d'atterrissages réussis puis le retour moyen.
Le trial retenu atterrit dans les 20 rollouts ; le GIF montre son rollout au meilleur retour.

![LunarLander rollout from a committed GPT-5.5 policy](../assets/lunarlander_policy.gif)

### Réglage d'un classificateur gradient boosting : probabilités de défaut de crédit

![Five-seed CPU-only GPT-5.5 context benchmark for UCI credit-default HGB tuning](../assets/credit_card.png)

Ce benchmark CPU-only règle huit paramètres d'entraînement d'un `HistGradientBoostingClassifier` sur le jeu UCI
[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) :
30 000 lignes, 23 variables et une cible de défaut le mois suivant. L'archive officielle est épinglée par SHA-256,
licenciée CC BY 4.0, et divisée une seule fois en 60% train, 20% validation et 20% test intact.
Toutes les méthodes utilisent la même partition, 20 trials et les seeds `0..4`. Les deux bras GPT-5.5 utilisent
high modeling effort, 20 trials d'historique de prompt, explicit reasoning et qualitative notes.

| méthode | log loss validation finale ↓ | log loss test retenu ↓ |
|---|---:|---:|
| Random | 0.433 | 0.425 |
| TPE | 0.430 | 0.422 |
| GP-BO | 0.430 | 0.423 |
| **GPT-5.5 w/ context** | **0.428** | **0.422** |
| GPT-5.5 w/o context | 0.433 | 0.427 |

Le contexte réduit la log loss validation finale de 1.13% et la log loss test de 1.23% par rapport au contrôle no-context.
GPT-5.5 a aussi une perte moyenne validation et test inférieure à Random, TPE et GP-BO. Comme la configuration retenue
a été choisie avec la validation et le test, le résultat test est une comparaison benchmark et non une estimation intacte de généralisation.

C'est un benchmark méthodologique, pas un système de décision de crédit en production. Le déploiement demanderait équité,
calibration, dérive, gouvernance et revue juridique.

Reproduire les artefacts benchmark :

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

## Guide d'utilisation

### Contrôles du prompt sampler

`effort` est transmis au flag reasoning-effort du backend CLI. Le prompt du harness est contrôlé séparément :

```python
oa.AgentSampler(
    backend="codex",
    effort="medium",
    history=5,
    explicit_reasoning=True,
    qualitative_notes=True,
)
```

Définissez `history=None` pour montrer tous les trials complétés/pruned. Utilisez
`explicit_reasoning=False` ou `qualitative_notes=False` pour raccourcir les réponses de l'agent.

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

L'agent pruner compare la courbe courante aux trials terminés et répond prune/keep ;
`loose` ne prune que les runs clairement faibles, tandis que `tight` prune plus agressivement.
Les erreurs d'agent ne prunent jamais un trial.

### Concurrence et studies distribuées

Définissez `max_concurrency` (défaut `1`) pour évaluer plusieurs trials à la fois, et utilisez un fichier
SQLite `storage` (`.db` / `.sqlite`) comme historique partagé sûr en concurrence :

```python
study = oa.create_study(
    sampler=oa.AgentSampler(backend="claude"),
    storage="study.db",        # SQLite -> safe for many workers; .json stays single-writer
    max_concurrency=8,         # up to 8 objectives run at once
)
study.optimize(objective, n_trials=100)
```

- **Dans un processus**, `max_concurrency` lance les objectifs dans un pool de threads. Les requêtes de sampling agent
  sont mises en file et sérialisées pour que chaque proposition voie l'historique du processus ; seuls les appels objectif sont parallèles.
- **Entre processus / machines**, pointez tous les workers vers le même `storage` SQLite. La base est le canal de communication :
  le mode WAL laisse chaque worker ajouter des résultats et lire l'historique sans conflit d'écriture.

Limites : les threads partagent le GIL, donc les objectifs pure-Python CPU-bound fonctionnent mieux dans des processus séparés avec SQLite partagé.
Les workers concurrents ne voient pas les points in-flight des autres et peuvent parfois explorer des régions proches.

### Mode skill (l'agent lit le code du projet)

Le package pip traite l'objectif comme une boîte noire. Le [skill optim-agent](../../SKILL.md) va plus loin :
chargé dans une session d'agent de code, l'agent lit d'abord le projet pour comprendre le rôle de chaque paramètre,
puis pilote la même boucle study via `study.ask(params)` / `study.tell(trial, value)`, avec le JSON de study comme historique.

```text
$skill-installer install https://github.com/Optim-Agent/optim-agent
```

Plugin Claude Code :

```bash
claude plugin marketplace add Optim-Agent/optim-agent
claude plugin install optim-agent@optim-agent
```

Plugin Codex :

```bash
codex plugin marketplace add Optim-Agent/optim-agent
codex plugin add optim-agent@optim-agent
```

```python
trial = study.ask({"threshold": 0.72, "budget": 80})
study.tell(trial, evaluate_system(**trial.params))
```

### Tests hors ligne

`AgentSampler(backend="mock")` est un substitut sans tokens qui fait un hill climbing autour du meilleur point,
utile pour tester les intégrations avant les appels agent.

## Dépannage

- **`claude` renvoie 401 dans une session agent** - les sessions imbriquées héritent de `ANTHROPIC_API_KEY` ;
  lancez avec `env -u ANTHROPIC_API_KEY` ou depuis un shell propre.
- **Un appel backend expire ou émet une sortie invalide** - le sampler avertit et revient à un point aléatoire pour ce trial ; la study continue.
- **OpenCode avec studies distribuées** - OpenCode currently does not support distributed computing
  in optim-agent ; utilisez le flux mono-processus ou un autre backend.

## Contribution

Développement local :

```bash
pip install -e ".[examples]"
pytest                     # runs tests/test_optim_agent.py
```

Ouvrez une issue pour discuter des changements importants avant un PR. Ajouter un nouveau backend agent signifie généralement
une petite fonction dans [`optim_agent/agent.py`](../../optim_agent/agent.py).

Le [`README.md`](../../README.md) anglais reste la source d'autorité pour les versions, valeurs benchmark et backends.

## Remerciements

- [Optuna](https://github.com/optuna/optuna) pour avoir popularisé l'interface Study/Trial, fourni la baseline TPE utilisée
  dans les exemples et benchmarks, et fixé un haut niveau pour les outils d'optimisation pratiques.
- [OpenCode](https://github.com/sst/opencode) pour l'accès aux modèles gratuits évalués dans les benchmarks de fonctions difficiles.

## Licence

[MIT](../../LICENSE)
