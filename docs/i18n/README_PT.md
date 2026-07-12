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

```bash
pip install optim-agent
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
    ),
    storage="study.json",
)
study.optimize(objective, n_trials=20)
print(study.best_value, study.best_params)
```

`context` é opcional, mas poderoso. Descreva o sistema e o significado de cada
parâmetro `suggest_*` para que o agente raciocine como um engenheiro de
algoritmos, em vez de apenas explorar pontos às cegas.

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

![Benchmarks de MNIST e CIFAR-10 com cinco sementes](../assets/classification_benchmarks.png)

As comparações usam o mesmo espaço de busca, orçamento e sementes para Random,
Optuna TPE e agentes com e sem contexto. Valores exatos, metodologia e comandos
de reprodução estão no
[README em inglês](../../README.md#benchmarks-agents-vs-tpe-and-random-search),
que é a referência oficial.

Outros exemplos:

- [Ajuste de inferência](../../examples/inference_tuning.py)
- [Ajuste walk-forward de sinais quantitativos](../../examples/quant_walk_forward.py)
- [Ajuste com scikit-learn](../../examples/sklearn_tuning.py)
- [MNIST](../../examples/mnist.py) / [CIFAR-10](../../examples/cifar10.py)

## Limitações atuais

- Hoje há suporte a um único objetivo. Para múltiplos objetivos, defina
  explicitamente uma utilidade escalar ou penalidades de restrição.
- Para avaliações muito baratas que permitem milhares de tentativas, TPE,
  processos gaussianos ou métodos evolutivos podem ser mais adequados.
- Para reprodutibilidade, fixe as sementes e preserve o estudo completo.

## Contribuição

Contribuições são bem-vindas. Discuta mudanças grandes em uma issue antes de
abrir um Pull Request. O [README em inglês](../../README.md) é a fonte oficial
para versões, resultados e backends compatíveis.

## Agradecimentos

- [Optuna](https://github.com/optuna/optuna), por popularizar a interface
  Study/Trial e fornecer a referência TPE usada nos exemplos e benchmarks.

## Licença

[MIT](../../LICENSE)
