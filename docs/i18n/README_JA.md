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

```bash
pip install optim-agent
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
    ),
    storage="study.json",
)
study.optimize(objective, n_trials=20)
print(study.best_value, study.best_params)
```

`context` は任意ですが重要です。システム全体または各 `suggest_*` パラメータに意味を
与えることで、エージェントは単なる点探索ではなくアルゴリズムエンジニアのように
推論できます。

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

![MNIST と CIFAR-10 の 5 シード比較](../assets/classification_benchmarks.png)

同一の探索空間、試行回数、シードで Random、Optuna TPE、コンテキストあり・なしの
エージェントを比較しています。正確な数値、方法、再現コマンドは更新の基準となる
[英語版 README](../../README.md#benchmarks-agents-vs-tpe-and-random-search) を参照してください。

追加例:

- [推論パラメータ調整](../../examples/inference_tuning.py)
- [量的シグナルのウォークフォワード調整](../../examples/quant_walk_forward.py)
- [scikit-learn モデル調整](../../examples/sklearn_tuning.py)
- [MNIST](../../examples/mnist.py) / [CIFAR-10](../../examples/cifar10.py)

## 現在の制約

- 現在は単一目的最適化です。多目的問題ではスカラー効用または制約ペナルティを
  明示してください。
- 非常に安価で数千回の試行が可能な問題では、TPE、GP、進化的手法の方が適する場合が
  あります。
- 再現性のため、シードを固定し、完全な study を保存してください。

## 貢献

コントリビューションを歓迎します。大きな変更は、Pull Request の前に Issue で
相談してください。最新のバージョン、ベンチマーク値、バックエンド一覧については
[英語版 README](../../README.md) が正本です。

## 謝辞

- Study/Trial インターフェースを普及させ、例とベンチマークで使用する TPE 基準を
  提供した [Optuna](https://github.com/optuna/optuna)。

## ライセンス

[MIT](../../LICENSE)
