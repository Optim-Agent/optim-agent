<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../assets/optim-agent-logo-dark.svg">
    <img alt="optim-agent" src="../assets/optim-agent-logo-light.svg" width="500">
  </picture>
</p>

<h1 align="center">optim-agent</h1>

<p align="center">
  <strong>코딩 에이전트를 활용한 에이전트형 시스템 최적화.</strong><br>
  알고리즘 엔지니어의 반복적인 파라미터 튜닝 작업을 자동화합니다.
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
  <strong>한국어</strong> |
  <a href="README_FR.md">Français</a> |
  <a href="README_DE.md">Deutsch</a> |
  <a href="README_ES.md">Español</a> |
  <a href="README_PT.md">Português</a> |
  <a href="README_RU.md">Русский</a>
</p>

optim-agent는 Claude Code / Codex / OpenCode가 코드를 읽고 trial을 제안하며
측정된 목적 결과를 기록해 실제 시스템 파라미터를 튜닝하게 합니다. 시스템이 설정 가능한
파라미터와 측정 가능한 목적을 제공할 때 사용합니다. 각 파라미터의 의미와 trial 이력이
보여주는 내용을 결합해 다음에 평가할 설정을 제안합니다. 목적 평가는 항상 authoritative합니다:
optim-agent는 값을 제안하고, 선언된 공간에 대해 검증하고, 결과를 기록하며, 에이전트 응답이
유효하지 않으면 안전한 샘플링으로 폴백합니다.

## 주요 특징

- **의미 기반 제안**: 익명 좌표만 다루지 않고 파라미터 의미, 연구 맥락, 관측 결과를
  바탕으로 추론합니다.
- **작은 실험 예산에 유용**: 평가 비용이 높고 고전적 대리 모델에 충분한 데이터가 없는
  상황에 적합합니다.
- **감사 가능성**: 설정, 결과, 상태, 맥락, 선택적 에이전트 추론을 JSON 또는 SQLite에
  저장합니다.
- **명확한 실행 경계**: 에이전트는 값만 제안하고, 탐색 공간이 검증하며, 목적 함수가
  결과를 결정합니다.

## 설치

Codex skill을 설치합니다.

```text
$skill-installer install https://github.com/Optim-Agent/optim-agent
```

Claude Code plugin을 설치합니다.

```bash
claude plugin marketplace add Optim-Agent/optim-agent && claude plugin install optim-agent@optim-agent
```

Python 패키지를 설치합니다.

```bash
# PyPI 안정 버전
python -m pip install optim-agent

# GitHub 최신 소스
python -m pip install "optim-agent @ git+https://github.com/Optim-Agent/optim-agent.git"
```

또한 인증이 완료된 `claude`, `codex`, `opencode` CLI 중 하나가 PATH에 있어야 합니다.

## 빠른 시작

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
        backend="codex",  # 또는 "claude" / "opencode"
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

`context`는 선택 사항이지만 강력합니다. 시스템 전체 또는 각 `suggest_*` 파라미터의
의미를 제공하면, 에이전트가 단순한 점 탐색기가 아니라 알고리즘 엔지니어처럼 추론할 수
있습니다.

## 스킬 모드

패키지 모드는 목적 함수를 블랙박스로 취급합니다. 루트의
[`SKILL.md`](../../SKILL.md)를 사용하면 현재 코딩 에이전트가 먼저 프로젝트를 읽고
파라미터 관계를 이해한 뒤 `study.ask(params)`와 `study.tell(trial, value)`로 같은
study를 진행합니다. 사용 중인 코딩 에이전트 환경에서 GitHub로부터 직접 불러올 수 있습니다.

```text
$skill-installer install https://github.com/Optim-Agent/optim-agent
```

## 적용 분야

| 분야 | 튜닝 가능한 예 | 목적 예시 |
|---|---|---|
| 모델 학습 | 학습률, 아키텍처, 증강, 정규화 | 품질, 연산량, 강건성 |
| 추론 및 서빙 | 양자화, 배칭, 디코딩, 캐시, 라우팅 | 품질, 지연 시간, 처리량, 비용 |
| 퀀트 연구 | 신호 윈도, 임계값, 리밸런싱, 위험 제어 | 워크포워드 수익, 낙폭, 회전율 |
| 강화학습 및 의사결정 | 목적 가중치, 탐색 일정, 정책 임계값 | 보상, 안전성, 샘플 효율 |
| 과학 워크플로 | 시뮬레이션 입력, 솔버, 실험 제어 | 적합도, 오차, 실행 시간, 자원 사용량 |
| 블랙박스 시스템 | 경계가 있는 범주형, 정수형, 연속형 설정 | 측정 가능한 모든 스칼라 값 |

## 벤치마크

### 문맥 없이 수학 함수 최적화: Branin-2D와 Ackley-5D

어려운 함수 에이전트에는 **제공된 작업 문맥이 없습니다**. 일반 이름 `x1...x5`, 수치 경계, trial 기록만 제공합니다. 모든 에이전트는 medium effort, 10 trials, 5개 seed를 사용하며 Random과 TPE는 동일한 baseline입니다.

#### 상위 에이전트

![No-context top-tier hard-function benchmark](../assets/hard_benchmarks_tier.png)

| 방법 | 평균 best Branin ↓ | 평균 best Ackley-5D ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| GPT-5.5 | 1.326 | 3.960 |
| **Opus-4.8** | **0.398** | **0.061** |
| Sonnet-5 | 3.850 | 0.143 |
| GLM-5.2 | 3.609 | 15.023 |

#### OpenCode 에이전트(무료)

![No-context free-model hard-function benchmark](../assets/hard_benchmarks_free.png)

| 방법 | 평균 best Branin ↓ | 평균 best Ackley-5D ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| Big-pickle | 4.734 | 15.951 |
| DeepSeek-V4-Flash | 4.410 | **4.608** |
| Nemotron-3-Ultra | 16.051 | 18.459 |
| **MiMo-v2.5** | **3.682** | 15.597 |

### ResNet 기반 이미지 분류기 튜닝: MNIST와 CIFAR-10

![MNIST 및 CIFAR-10 5개 시드 벤치마크](../assets/classification_benchmarks.png)

Random, Optuna TPE, **GPT-5.5 w/ context**, **GPT-5.5 w/o context**를 5개 seed(`0..4`)와 10 trials로 비교합니다. 두 GPT-5.5 조건은 `gpt-5.5`와 medium reasoning effort(`model_reasoning_effort=medium`)를 고정하며, **GPT-5.5 w/ context**만 study 목표와 16개 파라미터의 자연어 설명을 받습니다.

| 방법 | MNIST 누적 오류 ↓ | MNIST 최종 오류 ↓ | CIFAR-10 누적 오류 ↓ | CIFAR-10 최종 오류 ↓ |
|---|---:|---:|---:|---:|
| Random | 9.174 | 0.648% | 278.920 | 25.072% |
| TPE | 7.166 | 0.580% | 279.936 | 25.596% |
| **GPT-5.5 w/ context** | **5.668** | **0.506%** | **220.994** | **21.322%** |
| GPT-5.5 w/o context | 8.910 | 0.632% | 281.466 | 25.960% |

### Gradient Boosting 분류기 튜닝: 신용 부도 확률

![Five-seed CPU-only GPT-5.5 context benchmark for UCI credit-default HGB tuning](../assets/credit_card.png)

이 CPU-only 벤치마크는 UCI **Default of Credit Card Clients**(30,000행, 23개 특성, CC BY 4.0)에서 `HistGradientBoostingClassifier`의 8개 학습 파라미터를 튜닝합니다. 모든 방법은 같은 split, 20 trials, seed `0..4`를 사용합니다. 두 GPT-5.5 조건은 high modeling effort, 20개 trial history, explicit reasoning, qualitative notes를 사용합니다.

| 방법 | 최종 validation log loss ↓ | holdout test log loss ↓ |
|---|---:|---:|
| Random | 0.433 | 0.425 |
| TPE | 0.430 | 0.422 |
| **GPT-5.5 w/ context** | **0.428** | **0.422** |
| GPT-5.5 w/o context | 0.433 | 0.427 |

문맥은 대응 no-context control 대비 최종 validation log loss를 1.13%, test log loss를 1.23% 낮춥니다. GPT-5.5는 두 지표 모두에서 Random과 TPE를 앞섭니다. 구성 선택에 validation과 test loss를 모두 사용했으므로 test 결과는 독립적인 일반화 추정이 아니라 벤치마크 비교입니다. 이는 방법론 벤치마크이며 실제 신용 의사결정 시스템이 아닙니다.

추가 예제:

- [추론 파라미터 튜닝](../../examples/inference_tuning.py)
- [scikit-learn 모델 튜닝](../../examples/sklearn_tuning.py)
- [MNIST](../../examples/mnist.py) / [CIFAR-10](../../examples/cifar10.py)

## 현재 제한 사항

- 현재는 단일 목적 최적화만 지원합니다. 다목적 문제는 스칼라 효용이나 제약 페널티를
  명시해야 합니다.
- 평가가 매우 저렴하고 수천 번 시도할 수 있다면 TPE, GP 또는 진화 알고리즘이 더
  적합할 수 있습니다.
- 재현성을 위해 시드를 고정하고 전체 study를 저장하세요.

## 문제 해결

- **OpenCode와 분산 study**: OpenCode는 현재 optim-agent의
  `distributed computing` 워크플로를 지원하지 않습니다. 단일 프로세스 방식을
  사용하거나 분산 실행에는 다른 백엔드를 선택하세요.

## 기여

기여를 환영합니다. 큰 변경은 Pull Request 전에 Issue에서 논의해 주세요. 최신 버전,
벤치마크 수치, 백엔드 목록은 [영문 README](../../README.md)를 기준으로 합니다.

## 감사의 말

- Study/Trial 인터페이스를 널리 알리고 예제와 벤치마크의 TPE 기준선을 제공한
  [Optuna](https://github.com/optuna/optuna).
- 어려운 함수 벤치마크에서 평가한 무료 모델을 제공한
  [OpenCode](https://github.com/sst/opencode).

## 라이선스

[MIT](../../LICENSE)
