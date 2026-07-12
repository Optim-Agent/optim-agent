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

```bash
pip install optim-agent
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

![Benchmarks de MNIST y CIFAR-10 con cinco semillas](../assets/classification_benchmarks.png)

Las comparaciones usan el mismo espacio de búsqueda, presupuesto y semillas para
Random, Optuna TPE y los agentes con y sin contexto. Los valores exactos, la
metodología y los comandos de reproducción están en el
[README en inglés](../../README.md#benchmarks-agents-vs-tpe-and-random-search),
que es la referencia oficial.

Más ejemplos:

- [Ajuste de inferencia](../../examples/inference_tuning.py)
- [Ajuste walk-forward de señales cuantitativas](../../examples/quant_walk_forward.py)
- [Ajuste con scikit-learn](../../examples/sklearn_tuning.py)
- [MNIST](../../examples/mnist.py) / [CIFAR-10](../../examples/cifar10.py)

## Limitaciones actuales

- Actualmente admite un solo objetivo. Para varios objetivos, define de forma
  explícita una utilidad escalar o penalizaciones por restricciones.
- Si las evaluaciones son muy baratas y permiten miles de pruebas, TPE, los
  procesos gaussianos o los métodos evolutivos pueden ser más adecuados.
- Para poder reproducir los resultados, fija las semillas y conserva el estudio completo.

## Contribuir

Las contribuciones son bienvenidas. Comenta los cambios grandes en una issue
antes de abrir un Pull Request. El [README en inglés](../../README.md) es la
fuente oficial para versiones, resultados y backends compatibles.

## Agradecimientos

- [Optuna](https://github.com/optuna/optuna), por popularizar la interfaz
  Study/Trial y proporcionar la referencia TPE usada en ejemplos y benchmarks.

## Licencia

[MIT](../../LICENSE)
