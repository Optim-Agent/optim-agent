<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../assets/optim-agent-logo-dark.svg">
    <img alt="optim-agent" src="../assets/optim-agent-logo-light.svg" width="500">
  </picture>
</p>

<h1 align="center">optim-agent</h1>

<p align="center">
  <strong>Агентная оптимизация систем с помощью программирующих агентов.</strong><br>
  Автоматизация итеративной настройки параметров, которой занимается инженер по алгоритмам.
</p>

<p align="center">
  <a href="../../README.md">English</a> |
  <a href="README_ZH.md">简体中文</a> |
  <a href="README_JA.md">日本語</a> |
  <a href="README_KO.md">한국어</a> |
  <a href="README_FR.md">Français</a> |
  <a href="README_DE.md">Deutsch</a> |
  <a href="README_ES.md">Español</a> |
  <a href="README_PT.md">Português</a> |
  <strong>Русский</strong>
</p>

optim-agent использует Claude Code, Codex или OpenCode для оптимизации любой
системы с **настраиваемыми параметрами** и **измеримой целевой функцией**. Он
сопоставляет смысл параметров с историей испытаний и предлагает следующую
конфигурацию для оценки. Решение всегда остается за вашей целевой функцией:
некорректные ответы проверяются и заменяются безопасной выборкой.

## Зачем нужен optim-agent

- **Семантические предложения**: агент учитывает смысл параметров, контекст
  исследования и наблюдаемые результаты, а не только безымянные координаты.
- **Польза при малом бюджете**: подходит для дорогих оценок, когда классическим
  суррогатным моделям еще не хватает данных.
- **Аудируемость**: конфигурации, результаты, состояния, контекст и необязательное
  обоснование агента сохраняются в JSON или SQLite.
- **Ограниченное выполнение**: агент только предлагает значения; пространство
  поиска проверяет их, а целевая функция определяет результат.

## Установка

Выберите стабильную версию из PyPI или свежий исходный код с GitHub:

```bash
# Стабильная версия из PyPI
python -m pip install optim-agent

# Свежий исходный код с GitHub
python -m pip install "optim-agent @ git+https://github.com/Optim-Agent/optim-agent.git"
```

Также в PATH должен находиться авторизованный CLI `claude`, `codex` или `opencode`.

## Быстрый старт

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
        backend="codex",  # также "claude" / "opencode"
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

`context` необязателен, но очень полезен. Опишите систему и смысл каждого
параметра `suggest_*`, чтобы агент рассуждал как инженер по алгоритмам, а не как
слепой переборщик точек.

## Режим skill

Пакетный режим рассматривает цель как черный ящик. С корневым
[`SKILL.md`](../../SKILL.md) активный программирующий агент сначала читает
проект и понимает связи между параметрами, а затем ведет то же исследование
через `study.ask(params)` и `study.tell(trial, value)`. Загрузите skill
непосредственно с GitHub в активной среде программирующего агента:

```text
$skill-installer install https://github.com/Optim-Agent/optim-agent
```

## Области применения

| Область | Примеры параметров | Примеры целей |
|---|---|---|
| Обучение моделей | скорость обучения, архитектура, аугментация, регуляризация | качество, вычисления, устойчивость |
| Инференс и сервис | квантизация, батчинг, декодирование, кеш, маршрутизация | качество, задержка, пропускная способность, стоимость |
| Количественные исследования | окна сигналов, пороги, ребалансировка, контроль риска | walk-forward доходность, просадка, оборот |
| Обучение с подкреплением | веса целей, расписание исследования, пороги политики | доходность, безопасность, эффективность выборки |
| Научные процессы | входы симуляции, решатели, управление экспериментом | соответствие, ошибка, время, ресурсы |
| Системы «черного ящика» | любые ограниченные категориальные, целые или непрерывные настройки | любой измеримый скаляр |

## Бенчмарки

### Оптимизация математических функций без контекста: Branin-2D и Ackley-5D

Агенты для сложных функций получают **без предоставленного контекста задачи**: только общие имена `x1...x5`, числовые границы и историю trials. Все агенты используют medium effort, 10 trials и пять seed; Random и TPE остаются базовыми линиями.

#### Топовые агенты

![No-context top-tier hard-function benchmark](../assets/hard_benchmarks_tier.png)

| метод | средний лучший Branin ↓ | средний лучший Ackley-5D ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| GPT-5.5 | 1.326 | 3.960 |
| **Opus-4.8** | **0.398** | **0.061** |
| Sonnet-5 | 3.850 | 0.143 |
| GLM-5.2 | 3.609 | 15.023 |

#### Агенты OpenCode (бесплатные)

![No-context free-model hard-function benchmark](../assets/hard_benchmarks_free.png)

| метод | средний лучший Branin ↓ | средний лучший Ackley-5D ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| Big-pickle | 4.734 | 15.951 |
| DeepSeek-V4-Flash | 4.410 | **4.608** |
| Nemotron-3-Ultra | 16.051 | 18.459 |
| **MiMo-v2.5** | **3.682** | 15.597 |

### Настройка ResNet-классификаторов изображений: MNIST и CIFAR-10

![Бенчмарки MNIST и CIFAR-10 по пяти seed](../assets/classification_benchmarks.png)

Random, Optuna TPE, **GPT-5.5 w/ context** и **GPT-5.5 w/o context** сравниваются по пяти seed (`0..4`) и 10 trials. Обе конфигурации GPT-5.5 фиксируют `gpt-5.5` и medium reasoning effort (`model_reasoning_effort=medium`); только **GPT-5.5 w/ context** получает текстовые описания цели study и всех 16 параметров.

| метод | накопленная ошибка MNIST ↓ | финальная ошибка MNIST ↓ | накопленная ошибка CIFAR-10 ↓ | финальная ошибка CIFAR-10 ↓ |
|---|---:|---:|---:|---:|
| Random | 9.174 | 0.648% | 278.920 | 25.072% |
| TPE | 7.166 | 0.580% | 279.936 | 25.596% |
| **GPT-5.5 w/ context** | **5.668** | **0.506%** | **220.994** | **21.322%** |
| GPT-5.5 w/o context | 8.910 | 0.632% | 281.466 | 25.960% |

### Настройка gradient boosting classifier: вероятности кредитного дефолта

![Five-seed CPU-only GPT-5.5 context benchmark for UCI credit-default HGB tuning](../assets/credit_card.png)

Этот CPU-only benchmark настраивает восемь параметров обучения `HistGradientBoostingClassifier` на UCI **Default of Credit Card Clients** (30 000 строк, 23 признака, CC BY 4.0). Все методы используют одно и то же разбиение, 20 trials и seed `0..4`.

| метод | итоговый validation log loss ↓ | holdout test log loss ↓ |
|---|---:|---:|
| Random | 0.433 | 0.425 |
| TPE | 0.430 | **0.422** |
| **GPT-5.5 w/ context** | **0.428** | **0.422** |
| GPT-5.5 w/o context | 0.433 | 0.427 |

В выбранной конфигурации GPT-5.5 контекст снижает итоговый validation log loss на 1,16% и holdout test log loss на 1,15% относительно парного no-context контроля. Это методологический benchmark, а не промышленная система кредитных решений.

Другие примеры:

- [Настройка инференса](../../examples/inference_tuning.py)
- [Настройка scikit-learn](../../examples/sklearn_tuning.py)
- [MNIST](../../examples/mnist.py) / [CIFAR-10](../../examples/cifar10.py)

## Текущие ограничения

- Сейчас поддерживается одна цель. Для нескольких целей явно задайте скалярную
  полезность или штрафы за нарушение ограничений.
- Для очень дешевых оценок с тысячами испытаний лучше могут подойти TPE,
  гауссовские процессы или эволюционные методы.
- Для воспроизводимости фиксируйте seed и сохраняйте исследование целиком.

## Устранение неполадок

- **OpenCode и распределенные исследования**: OpenCode пока не поддерживает
  workflow `distributed computing` в optim-agent. Используйте один процесс или
  другой бэкенд для распределенных запусков.

## Участие в разработке

Мы приветствуем вклад в проект. Крупные изменения стоит сначала обсудить в issue,
а затем открывать Pull Request. [Английский README](../../README.md) является
основным источником версий, результатов и списка поддерживаемых бэкендов.

## Благодарности

- [Optuna](https://github.com/optuna/optuna) за популяризацию интерфейса
  Study/Trial и базовую линию TPE, используемую в примерах и бенчмарках.
- [OpenCode](https://github.com/sst/opencode) за доступ к бесплатным моделям,
  протестированным в бенчмарках сложных функций.

## Лицензия

[MIT](../../LICENSE)
