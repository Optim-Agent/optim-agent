<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../assets/optim-agent-logo-dark.svg">
    <img alt="optim-agent" src="../assets/optim-agent-logo-light.svg" width="500">
  </picture>
</p>

<h1 align="center">optim-agent</h1>

<p align="center">
  <strong>Optimización agéntica de sistemas con agentes de programación.</strong><br>
  Automatiza el ajuste iterativo de parámetros que realiza un ingeniero de algoritmos.
</p>

<p align="center">
  <a href="../../README.md">English</a> |
  <a href="README_ZH.md">简体中文</a> |
  <a href="README_JA.md">日本語</a> |
  <a href="README_KO.md">한국어</a> |
  <a href="README_FR.md">Français</a> |
  <a href="README_DE.md">Deutsch</a> |
  <strong>Español</strong> |
  <a href="README_PT.md">Português</a> |
  <a href="README_RU.md">Русский</a>
</p>

optim-agent usa Claude Code, Codex u OpenCode para optimizar cualquier sistema
que exponga **parámetros configurables** y un **objetivo medible**. Combina el
significado de cada parámetro con el historial de pruebas y propone la siguiente
configuración que se debe evaluar. Tu función objetivo conserva la autoridad:
las respuestas no válidas se verifican y se sustituyen por un muestreo seguro.

## Por qué optim-agent

- **Propuestas semánticas**: el agente razona sobre el significado de los
  parámetros, el contexto del estudio y los resultados observados.
- **Ventaja con presupuestos pequeños**: útil cuando cada evaluación es cara y
  los modelos sustitutos clásicos todavía tienen pocos datos.
- **Auditable**: conserva configuraciones, resultados, estados, contexto y el
  razonamiento opcional del agente en JSON o SQLite.
- **Ejecución acotada**: el agente solo propone valores; el espacio de búsqueda
  los valida y la función objetivo decide el resultado.

## Instalación

Elige la versión estable de PyPI o el código más reciente de GitHub:

```bash
# Versión estable de PyPI
python -m pip install optim-agent

# Código más reciente de GitHub
python -m pip install "optim-agent @ git+https://github.com/Optim-Agent/optim-agent.git"
```

También necesitas un CLI `claude`, `codex` u `opencode` autenticado en tu PATH.

## Inicio rápido

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
        backend="codex",  # también "claude" / "opencode"
        effort="high",
        context="maximize quality under a strict operating-cost budget",
    ),
    storage="study.json",
)
study.optimize(objective, n_trials=20)
print(study.best_value, study.best_params)
```

`context` es opcional pero potente. Describe el sistema completo y el sentido de
cada parámetro `suggest_*` para que el agente razone como un ingeniero de
algoritmos y no como un explorador ciego de puntos.

## Modo skill

El modo paquete trata el objetivo como una caja negra. Con el
[`SKILL.md`](../../SKILL.md) de la raíz, el agente de programación activo lee
primero el proyecto y entiende las relaciones entre parámetros; después dirige
el mismo estudio mediante `study.ask(params)` y `study.tell(trial, value)`.
Instala el skill directamente desde GitHub en Codex:

```text
$skill-installer install https://github.com/Optim-Agent/optim-agent
```

## Ámbitos de aplicación

| Área | Ejemplos de parámetros | Ejemplos de objetivos |
|---|---|---|
| Entrenamiento de modelos | tasas de aprendizaje, arquitecturas, aumento, regularización | calidad, cómputo, robustez |
| Inferencia y servicio | cuantización, lotes, decodificación, caché, enrutamiento | calidad, latencia, rendimiento, coste |
| Investigación cuantitativa | ventanas de señal, umbrales, rebalanceo, control de riesgo | retorno walk-forward, drawdown, rotación |
| Aprendizaje por refuerzo | pesos de objetivos, exploración, umbrales de política | retorno, seguridad, eficiencia muestral |
| Flujos científicos | entradas de simulación, solvers, controles experimentales | ajuste, error, tiempo, recursos |
| Sistemas de caja negra | cualquier configuración categórica, entera o continua acotada | cualquier escalar medible |

## Benchmarks

### Optimización de funciones matemáticas sin contexto: Branin-2D y Ackley-5D

Los agentes de funciones difíciles reciben **sin contexto de tarea suministrado**: solo nombres genéricos `x1...x5`, límites numéricos e historial de pruebas. Todos usan esfuerzo medium, 10 pruebas y cinco semillas; Random y TPE son las mismas líneas base.

#### Agentes de primer nivel

![No-context top-tier hard-function benchmark](../assets/hard_benchmarks_tier.png)

| método | mejor Branin medio ↓ | mejor Ackley-5D medio ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| GPT-5.5 | 1.326 | 3.960 |
| **Opus-4.8** | **0.398** | **0.061** |
| Sonnet-5 | 3.850 | 0.143 |
| GLM-5.2 | 3.609 | 15.023 |

#### Agentes OpenCode (gratis)

![No-context free-model hard-function benchmark](../assets/hard_benchmarks_free.png)

| método | mejor Branin medio ↓ | mejor Ackley-5D medio ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| Big-pickle | 4.734 | 15.951 |
| DeepSeek-V4-Flash | 4.410 | **4.608** |
| Nemotron-3-Ultra | 16.051 | 18.459 |
| **MiMo-v2.5** | **3.682** | 15.597 |

### Ajuste de clasificadores de imagen basados en ResNet: MNIST y CIFAR-10

![Benchmarks de MNIST y CIFAR-10 con cinco semillas](../assets/classification_benchmarks.png)

Random, Optuna TPE, **GPT-5.5 w/ context** y **GPT-5.5 w/o context** se comparan con cinco semillas (`0..4`) y 10 pruebas. Ambas condiciones GPT-5.5 fijan `gpt-5.5` con esfuerzo de razonamiento medium (`model_reasoning_effort=medium`); solo **GPT-5.5 w/ context** recibe texto natural sobre el objetivo y los 16 parámetros.

| método | error acumulado MNIST ↓ | error final MNIST ↓ | error acumulado CIFAR-10 ↓ | error final CIFAR-10 ↓ |
|---|---:|---:|---:|---:|
| Random | 9.174 | 0.648% | 278.920 | 25.072% |
| TPE | 7.166 | 0.580% | 279.936 | 25.596% |
| **GPT-5.5 w/ context** | **5.668** | **0.506%** | **220.994** | **21.322%** |
| GPT-5.5 w/o context | 8.910 | 0.632% | 281.466 | 25.960% |

### Ajuste de clasificador gradient boosting: probabilidades de impago crediticio

![Five-seed CPU-only GPT-5.5 context benchmark for UCI credit-default HGB tuning](../assets/credit_card.png)

Este benchmark solo con CPU ajusta ocho parámetros de entrenamiento de un `HistGradientBoostingClassifier` sobre UCI **Default of Credit Card Clients** (30.000 filas, 23 variables, CC BY 4.0). Todos los métodos usan la misma partición, 20 pruebas y semillas `0..4`.

| método | log loss final de validación ↓ | log loss en test holdout ↓ |
|---|---:|---:|
| Random | 0.433 | 0.425 |
| TPE | 0.430 | **0.422** |
| **GPT-5.5 w/ context** | **0.428** | **0.422** |
| GPT-5.5 w/o context | 0.433 | 0.427 |

Con la configuración GPT-5.5 seleccionada, el contexto reduce el log loss final de validación un 1,16% y el test holdout un 1,15% frente al control sin contexto. Es un benchmark metodológico, no un sistema de decisión crediticia para producción.

Más ejemplos:

- [Ajuste de inferencia](../../examples/inference_tuning.py)
- [Ajuste con scikit-learn](../../examples/sklearn_tuning.py)
- [MNIST](../../examples/mnist.py) / [CIFAR-10](../../examples/cifar10.py)

## Limitaciones actuales

- Actualmente admite un solo objetivo. Para varios objetivos, define de forma
  explícita una utilidad escalar o penalizaciones por restricciones.
- Si las evaluaciones son muy baratas y permiten miles de pruebas, TPE, los
  procesos gaussianos o los métodos evolutivos pueden ser más adecuados.
- Para poder reproducir los resultados, fija las semillas y conserva el estudio completo.

## Solución de problemas

- **OpenCode y estudios distribuidos**: OpenCode no admite actualmente el flujo
  de `distributed computing` de optim-agent. Usa el flujo de un solo proceso o
  elige otro backend para las ejecuciones distribuidas.

## Contribuir

Las contribuciones son bienvenidas. Comenta los cambios grandes en una issue
antes de abrir un Pull Request. El [README en inglés](../../README.md) es la
fuente oficial para versiones, resultados y backends compatibles.

## Agradecimientos

- [Optuna](https://github.com/optuna/optuna), por popularizar la interfaz
  Study/Trial y proporcionar la referencia TPE usada en ejemplos y benchmarks.
- [OpenCode](https://github.com/sst/opencode), por dar acceso a los modelos
  gratuitos evaluados en los benchmarks de funciones difíciles.

## Licencia

[MIT](../../LICENSE)
