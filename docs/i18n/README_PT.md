<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../assets/optim-agent-logo-dark.svg">
    <img alt="optim-agent" src="../assets/optim-agent-logo-light.svg" width="500">
  </picture>
</p>

<h1 align="center">optim-agent</h1>

<p align="center">
  <strong>Otimização agêntica de sistemas com agentes de programação.</strong><br>
  Automatize o ajuste iterativo de parâmetros feito por um engenheiro de algoritmos.
</p>

<p align="center">
  <a href="../../README.md">English</a> |
  <a href="README_ZH.md">简体中文</a> |
  <a href="README_JA.md">日本語</a> |
  <a href="README_KO.md">한국어</a> |
  <a href="README_FR.md">Français</a> |
  <a href="README_DE.md">Deutsch</a> |
  <a href="README_ES.md">Español</a> |
  <strong>Português</strong> |
  <a href="README_RU.md">Русский</a>
</p>

optim-agent usa Claude Code, Codex ou OpenCode para otimizar qualquer sistema
que exponha **parâmetros configuráveis** e um **objetivo mensurável**. Ele une o
significado de cada parâmetro ao histórico de tentativas e propõe a próxima
configuração a avaliar. Sua função objetivo continua sendo a autoridade:
respostas inválidas são verificadas e substituídas por amostragem segura.

## Por que usar optim-agent

- **Propostas semânticas**: o agente raciocina sobre o significado dos
  parâmetros, o contexto do estudo e os resultados observados.
- **Vantagem com poucos testes**: útil quando cada avaliação é cara e modelos
  substitutos clássicos ainda têm poucos dados.
- **Auditável**: configurações, resultados, estados, contexto e raciocínio
  opcional do agente ficam registrados em JSON ou SQLite.
- **Execução limitada**: o agente apenas propõe valores; o espaço de busca os
  valida e a função objetivo decide o resultado.

## Instalação

Escolha a versão estável do PyPI ou o código mais recente do GitHub:

```bash
# Versão estável do PyPI
python -m pip install optim-agent

# Código mais recente do GitHub
python -m pip install "optim-agent @ git+https://github.com/Optim-Agent/optim-agent.git"
```

Também é necessário ter um CLI `claude`, `codex` ou `opencode` autenticado no PATH.

## Início rápido

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

`context` é opcional, mas poderoso. Descreva o sistema e o significado de cada
parâmetro `suggest_*` para que o agente raciocine como um engenheiro de
algoritmos, em vez de apenas explorar pontos às cegas.

## Modo skill

O modo pacote trata o objetivo como uma caixa-preta. Com o
[`SKILL.md`](../../SKILL.md) na raiz, o agente de programação ativo primeiro lê
o projeto e entende as relações entre parâmetros; depois conduz o mesmo estudo
com `study.ask(params)` e `study.tell(trial, value)`. Carregue o skill diretamente
do GitHub no ambiente ativo do agente de programação:

```text
$skill-installer install https://github.com/Optim-Agent/optim-agent
```

## Onde se aplica

| Área | Exemplos de parâmetros | Exemplos de objetivos |
|---|---|---|
| Treinamento de modelos | taxas de aprendizado, arquiteturas, aumento, regularização | qualidade, computação, robustez |
| Inferência e serving | quantização, lotes, decodificação, cache, roteamento | qualidade, latência, vazão, custo |
| Pesquisa quantitativa | janelas de sinal, limiares, rebalanceamento, controle de risco | retorno walk-forward, drawdown, giro |
| Aprendizado por reforço | pesos de objetivos, exploração, limiares de política | retorno, segurança, eficiência amostral |
| Fluxos científicos | entradas de simulação, solvers, controles experimentais | ajuste, erro, tempo, recursos |
| Sistemas caixa-preta | qualquer configuração categórica, inteira ou contínua limitada | qualquer escalar mensurável |

## Benchmarks

### Otimizando funções matemáticas sem contexto: Branin-2D e Ackley-5D

Os agentes de funções difíceis recebem **nenhum contexto de tarefa fornecido**: apenas nomes genéricos `x1...x5`, limites numéricos e histórico de tentativas. Todos usam effort medium, 10 tentativas e cinco sementes; Random e TPE são as mesmas linhas de base.

#### Agentes de primeira linha

![No-context top-tier hard-function benchmark](../assets/hard_benchmarks_tier.png)

| método | melhor Branin médio ↓ | melhor Ackley-5D médio ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| GPT-5.5 | 1.326 | 3.960 |
| **Opus-4.8** | **0.398** | **0.061** |
| Sonnet-5 | 3.850 | 0.143 |
| GLM-5.2 | 3.609 | 15.023 |

#### Agentes OpenCode (gratuitos)

![No-context free-model hard-function benchmark](../assets/hard_benchmarks_free.png)

| método | melhor Branin médio ↓ | melhor Ackley-5D médio ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| Big-pickle | 4.734 | 15.951 |
| DeepSeek-V4-Flash | 4.410 | **4.608** |
| Nemotron-3-Ultra | 16.051 | 18.459 |
| **MiMo-v2.5** | **3.682** | 15.597 |

### Ajustando classificadores de imagem baseados em ResNet: MNIST e CIFAR-10

![Benchmarks de MNIST e CIFAR-10 com cinco sementes](../assets/classification_benchmarks.png)

Random, Optuna TPE, **GPT-5.5 w/ context** e **GPT-5.5 w/o context** são comparados em cinco sementes (`0..4`) e 10 tentativas. Ambas as condições GPT-5.5 fixam `gpt-5.5` com esforço de raciocínio medium (`model_reasoning_effort=medium`); só **GPT-5.5 w/ context** recebe texto natural sobre o objetivo e os 16 parâmetros.

| método | erro acumulado MNIST ↓ | erro final MNIST ↓ | erro acumulado CIFAR-10 ↓ | erro final CIFAR-10 ↓ |
|---|---:|---:|---:|---:|
| Random | 9.174 | 0.648% | 278.920 | 25.072% |
| TPE | 7.166 | 0.580% | 279.936 | 25.596% |
| **GPT-5.5 w/ context** | **5.668** | **0.506%** | **220.994** | **21.322%** |
| GPT-5.5 w/o context | 8.910 | 0.632% | 281.466 | 25.960% |

### Ajustando classificador gradient boosting: probabilidades de inadimplência

![Five-seed CPU-only GPT-5.5 context benchmark for UCI credit-default HGB tuning](../assets/credit_card.png)

Este benchmark somente CPU ajusta oito parâmetros de treino de um `HistGradientBoostingClassifier` no UCI **Default of Credit Card Clients** (30.000 linhas, 23 atributos, CC BY 4.0). Todos os métodos usam a mesma divisão, 20 tentativas e sementes `0..4`.

| método | log loss final de validação ↓ | log loss no holdout de teste ↓ |
|---|---:|---:|
| Random | 0.433 | 0.425 |
| TPE | 0.430 | **0.422** |
| **GPT-5.5 w/ context** | **0.428** | **0.422** |
| GPT-5.5 w/o context | 0.433 | 0.427 |

Com a configuração GPT-5.5 selecionada, o contexto reduz o log loss final de validação em 1,16% e o log loss do holdout de teste em 1,15% contra o controle sem contexto. É um benchmark metodológico, não um sistema de decisão de crédito em produção.

Outros exemplos:

- [Ajuste de inferência](../../examples/inference_tuning.py)
- [Ajuste com scikit-learn](../../examples/sklearn_tuning.py)
- [MNIST](../../examples/mnist.py) / [CIFAR-10](../../examples/cifar10.py)

## Limitações atuais

- Hoje há suporte a um único objetivo. Para múltiplos objetivos, defina
  explicitamente uma utilidade escalar ou penalidades de restrição.
- Para avaliações muito baratas que permitem milhares de tentativas, TPE,
  processos gaussianos ou métodos evolutivos podem ser mais adequados.
- Para reprodutibilidade, fixe as sementes e preserve o estudo completo.

## Solução de problemas

- **OpenCode e estudos distribuídos**: o OpenCode atualmente não oferece suporte
  ao fluxo de `distributed computing` do optim-agent. Use o fluxo de processo
  único ou outro backend para execuções distribuídas.

## Contribuição

Contribuições são bem-vindas. Discuta mudanças grandes em uma issue antes de
abrir um Pull Request. O [README em inglês](../../README.md) é a fonte oficial
para versões, resultados e backends compatíveis.

## Agradecimentos

- [Optuna](https://github.com/optuna/optuna), por popularizar a interface
  Study/Trial e fornecer a referência TPE usada nos exemplos e benchmarks.
- [OpenCode](https://github.com/sst/opencode), por dar acesso aos modelos
  gratuitos avaliados nos benchmarks de funções difíceis.

## Licença

[MIT](../../LICENSE)
