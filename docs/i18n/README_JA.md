<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../assets/optim-agent-logo-dark.svg">
    <img alt="optim-agent" src="../assets/optim-agent-logo-light.svg" width="500">
  </picture>
</p>

<h1 align="center">optim-agent</h1>

<p align="center">
  <strong>コーディングエージェントによるエージェント型システム最適化。</strong><br>
  アルゴリズムエンジニアが行う反復的なパラメータ調整を自動化します。
</p>

<p align="center">
  <a href="../../README.md">English</a> |
  <a href="README_ZH.md">简体中文</a> |
  <strong>日本語</strong> |
  <a href="README_KO.md">한국어</a> |
  <a href="README_FR.md">Français</a> |
  <a href="README_DE.md">Deutsch</a> |
  <a href="README_ES.md">Español</a> |
  <a href="README_PT.md">Português</a> |
  <a href="README_RU.md">Русский</a>
</p>

optim-agent は Claude Code、Codex、OpenCode を利用し、**設定可能なパラメータ**と
**測定可能な目的関数**を持つあらゆるシステムを最適化します。パラメータの意味と
試行履歴の両方を読み取り、次に評価すべき設定を提案します。目的関数が常に最終的な
判断基準であり、無効な提案は検証後に安全なサンプリングへフォールバックします。

## 特長

- **意味を理解した提案**: 匿名の座標ではなく、パラメータの意味、研究コンテキスト、
  観測結果をもとに推論します。
- **少ない試行予算に強い**: 1 回の評価が高価で、古典的な代理モデルに十分なデータが
  まだない場面に適しています。
- **監査可能**: 設定、結果、状態、コンテキスト、任意のエージェント推論を JSON または
  SQLite に保存します。
- **境界のある実行**: エージェントは値を提案するだけで、探索空間が検証し、目的関数が
  成否を決めます。

## インストール

PyPI の安定版または GitHub の最新ソースのどちらか一方を選びます。

```bash
# PyPI の安定版
python -m pip install optim-agent

# GitHub の最新ソース
python -m pip install "optim-agent @ git+https://github.com/Optim-Agent/optim-agent.git"
```

さらに、認証済みの `claude`、`codex`、`opencode` CLI のいずれかを PATH に用意します。

## クイックスタート

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
        backend="codex",  # "claude" / "opencode" も利用可能
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

`context` は任意ですが重要です。システム全体または各 `suggest_*` パラメータに意味を
与えることで、エージェントは単なる点探索ではなくアルゴリズムエンジニアのように
推論できます。

## スキルモード

パッケージモードは目的関数をブラックボックスとして扱います。ルートの
[`SKILL.md`](../../SKILL.md) を使うと、現在のコーディングエージェントが先に
プロジェクトを読み、パラメータの関係を理解してから `study.ask(params)` と
`study.tell(trial, value)` で同じ study を進めます。利用中のコーディングエージェント環境に
GitHub から直接読み込めます。

```text
$skill-installer install https://github.com/Optim-Agent/optim-agent
```

## 適用範囲

| 分野 | 調整できる例 | 目的の例 |
|---|---|---|
| モデル学習 | 学習率、アーキテクチャ、拡張、正則化 | 品質、計算量、頑健性 |
| 推論・配信 | 量子化、バッチ、デコード、キャッシュ、ルーティング | 品質、遅延、スループット、コスト |
| クオンツ研究 | シグナル窓、閾値、リバランス、リスク制御 | ウォークフォワード収益、ドローダウン、回転率 |
| 強化学習・意思決定 | 目的重み、探索スケジュール、方策閾値 | リターン、安全性、サンプル効率 |
| 科学ワークフロー | シミュレーション入力、ソルバー、実験制御 | 適合度、誤差、実行時間、資源使用量 |
| ブラックボックス系 | 有界なカテゴリ・整数・連続設定 | 測定可能な任意のスカラー値 |

## ベンチマーク

### 文脈なしで数理関数を最適化: Branin-2D と Ackley-5D

難関関数のエージェントには **タスク文脈を与えません**。汎用名 `x1...x5`、数値境界、試行履歴だけを渡します。すべてのエージェントは medium effort、10 試行、5 シードで、Random と TPE は同じベースラインです。

#### 上位エージェント

![No-context top-tier hard-function benchmark](../assets/hard_benchmarks_tier.png)

| 手法 | 平均 best Branin ↓ | 平均 best Ackley-5D ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| GPT-5.5 | 1.326 | 3.960 |
| **Opus-4.8** | **0.398** | **0.061** |
| Sonnet-5 | 3.850 | 0.143 |
| GLM-5.2 | 3.609 | 15.023 |

#### OpenCode エージェント（無料）

![No-context free-model hard-function benchmark](../assets/hard_benchmarks_free.png)

| 手法 | 平均 best Branin ↓ | 平均 best Ackley-5D ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| Big-pickle | 4.734 | 15.951 |
| DeepSeek-V4-Flash | 4.410 | **4.608** |
| Nemotron-3-Ultra | 16.051 | 18.459 |
| **MiMo-v2.5** | **3.682** | 15.597 |

### ResNet ベース画像分類器の調整: MNIST と CIFAR-10

![MNIST と CIFAR-10 の 5 シード比較](../assets/classification_benchmarks.png)

Random、Optuna TPE、**GPT-5.5 w/ context**、**GPT-5.5 w/o context** を 5 シード（`0..4`）と 10 試行で比較します。両方の GPT-5.5 条件は `gpt-5.5` と medium reasoning effort（`model_reasoning_effort=medium`）を固定し、**GPT-5.5 w/ context** だけが study 目的と 16 パラメータの自然言語説明を受け取ります。

| 手法 | MNIST 累積エラー ↓ | MNIST 最終エラー ↓ | CIFAR-10 累積エラー ↓ | CIFAR-10 最終エラー ↓ |
|---|---:|---:|---:|---:|
| Random | 9.174 | 0.648% | 278.920 | 25.072% |
| TPE | 7.166 | 0.580% | 279.936 | 25.596% |
| **GPT-5.5 w/ context** | **5.668** | **0.506%** | **220.994** | **21.322%** |
| GPT-5.5 w/o context | 8.910 | 0.632% | 281.466 | 25.960% |

### Gradient Boosting 分類器の調整: クレジットデフォルト確率

![Five-seed CPU-only GPT-5.5 context benchmark for UCI credit-default HGB tuning](../assets/credit_card.png)

この CPU-only ベンチマークは、UCI **Default of Credit Card Clients**（30,000 行、23 特徴、CC BY 4.0）で `HistGradientBoostingClassifier` の 8 個の学習パラメータを調整します。すべての手法は同じ分割、20 試行、シード `0..4` を使います。

| 手法 | 最終 validation log loss ↓ | holdout test log loss ↓ |
|---|---:|---:|
| Random | 0.433 | 0.425 |
| TPE | 0.430 | **0.422** |
| **GPT-5.5 w/ context** | **0.428** | **0.422** |
| GPT-5.5 w/o context | 0.433 | 0.427 |

選択した GPT-5.5 構成では、文脈により対応する no-context 対照より final validation log loss が 1.16%、holdout test log loss が 1.15% 低下します。これは方法論ベンチマークであり、本番の与信判断システムではありません。

## 現在の制約

- 現在は単一目的最適化です。多目的問題ではスカラー効用または制約ペナルティを
  明示してください。
- 非常に安価で数千回の試行が可能な問題では、TPE、GP、進化的手法の方が適する場合が
  あります。
- 再現性のため、シードを固定し、完全な study を保存してください。

## トラブルシューティング

- **OpenCode と分散 study**: OpenCode は現在、optim-agent の
  `distributed computing` ワークフローをサポートしていません。単一プロセスを
  使用するか、分散実行には別のバックエンドを選んでください。

## 貢献

コントリビューションを歓迎します。大きな変更は、Pull Request の前に Issue で
相談してください。最新のバージョン、ベンチマーク値、バックエンド一覧については
[英語版 README](../../README.md) が正本です。

## 謝辞

- Study/Trial インターフェースを普及させ、例とベンチマークで使用する TPE 基準を
  提供した [Optuna](https://github.com/optuna/optuna)。
- 困難関数ベンチマークで評価した無料モデルを提供した
  [OpenCode](https://github.com/sst/opencode)。

## ライセンス

[MIT](../../LICENSE)
