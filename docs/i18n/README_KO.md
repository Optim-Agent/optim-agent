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

optim-agent는 Claude Code, Codex 또는 OpenCode를 사용해 **설정 가능한 파라미터**와
**측정 가능한 목적 함수**를 가진 모든 시스템을 최적화합니다. 각 파라미터의 의미와
실험 이력을 함께 이해한 뒤 다음에 평가할 설정을 제안합니다. 최종 판단은 항상 사용자의
목적 함수가 내리며, 잘못된 제안은 검증 후 안전한 샘플링으로 대체됩니다.

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

```bash
pip install optim-agent
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
    ),
    storage="study.json",
)
study.optimize(objective, n_trials=20)
print(study.best_value, study.best_params)
```

`context`는 선택 사항이지만 강력합니다. 시스템 전체 또는 각 `suggest_*` 파라미터의
의미를 제공하면, 에이전트가 단순한 점 탐색기가 아니라 알고리즘 엔지니어처럼 추론할 수
있습니다.

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

![MNIST 및 CIFAR-10 5개 시드 벤치마크](../assets/classification_benchmarks.png)

동일한 탐색 공간, 실험 횟수, 시드에서 Random, Optuna TPE, 맥락이 있는 에이전트와
없는 에이전트를 비교합니다. 정확한 수치, 방법론, 재현 명령은 기준 문서인
[영문 README](../../README.md#benchmarks-agents-vs-tpe-and-random-search)를 확인하세요.

추가 예제:

- [추론 파라미터 튜닝](../../examples/inference_tuning.py)
- [퀀트 신호 워크포워드 튜닝](../../examples/quant_walk_forward.py)
- [scikit-learn 모델 튜닝](../../examples/sklearn_tuning.py)
- [MNIST](../../examples/mnist.py) / [CIFAR-10](../../examples/cifar10.py)

## 현재 제한 사항

- 현재는 단일 목적 최적화만 지원합니다. 다목적 문제는 스칼라 효용이나 제약 페널티를
  명시해야 합니다.
- 평가가 매우 저렴하고 수천 번 시도할 수 있다면 TPE, GP 또는 진화 알고리즘이 더
  적합할 수 있습니다.
- 재현성을 위해 시드를 고정하고 전체 study를 저장하세요.

## 기여

기여를 환영합니다. 큰 변경은 Pull Request 전에 Issue에서 논의해 주세요. 최신 버전,
벤치마크 수치, 백엔드 목록은 [영문 README](../../README.md)를 기준으로 합니다.

## 감사의 말

- Study/Trial 인터페이스를 널리 알리고 예제와 벤치마크의 TPE 기준선을 제공한
  [Optuna](https://github.com/optuna/optuna).

## 라이선스

[MIT](../../LICENSE)
