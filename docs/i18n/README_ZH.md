<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../assets/optim-agent-logo-dark.svg">
    <img alt="optim-agent" src="../assets/optim-agent-logo-light.svg" width="500">
  </picture>
</p>

<h1 align="center">optim-agent</h1>

<p align="center">
  <strong>让大模型编程智能体成为你的超参数优化器。</strong><br>
  面向通用黑盒优化、机器学习训练、量化研究和 AI 推理配置的轻量工具。
</p>

<p align="center">
  <a href="../../README.md">English</a> |
  <strong>简体中文</strong> |
  <a href="README_JA.md">日本語</a> |
  <a href="README_KO.md">한국어</a> |
  <a href="README_FR.md">Français</a> |
  <a href="README_DE.md">Deutsch</a> |
  <a href="README_ES.md">Español</a> |
  <a href="README_PT.md">Português</a> |
  <a href="README_RU.md">Русский</a>
</p>

optim-agent 不使用内置的贝叶斯代理模型来决定下一个采样点，而是调用已经
安装并登录的 Claude Code、Codex 或 OpenCode CLI。智能体既能分析历史试验
数值，也能理解“学习率”“回看窗口”“量化精度”等参数的语义。

核心包只有 Python 标准库依赖。无需额外托管服务，也无需把模型 API Key
交给 optim-agent；它复用本机 CLI 的认证状态。

## 为什么使用 optim-agent

- **小预算优化**：适合每次试验昂贵、总预算只有十几到几十次的场景。
- **语义上下文**：通过 study 级和参数级 `context` 提供领域先验。
- **可审计**：保留参数、目标值、状态、中间指标和智能体备注。
- **有边界的自动化**：智能体只提出参数，搜索空间负责校验，目标函数决定结果。
- **熟悉的接口**：提供类似 Optuna 的 `create_study`、`suggest_*`、
  `optimize`、`ask` 和 `tell`。
- **安全降级**：CLI 超时、失败或返回无效 JSON 时，该次试验自动回退到随机采样。

## 安装

从 PyPI 安装稳定版本，或从 GitHub 安装最新源码；二者任选其一：

```bash
# PyPI 稳定版本
python -m pip install optim-agent

# GitHub 最新源码
python -m pip install "optim-agent @ git+https://github.com/Optim-Agent/optim-agent.git"
```

若要使用真实智能体，请至少安装并登录一个 CLI：
[Claude Code](https://docs.anthropic.com/en/docs/claude-code)、
[Codex](https://github.com/openai/codex) 或
[OpenCode](https://github.com/sst/opencode)。

开发和示例依赖按需安装：

```bash
pip install -e ".[examples,dev]"       # 基准、绘图和测试
pip install -e ".[ml]"                 # scikit-learn 示例
pip install -e ".[vision]"             # MNIST / CIFAR-10 示例
```

## 五分钟快速开始

先使用免费的 `mock` 后端检查目标函数和搜索空间，不消耗模型调用：

```python
import optim_agent as oa

def objective(trial):
    lr = trial.suggest_float(
        "lr", 1e-5, 1e-1, log=True,
        context="图像分类器的学习率",
    )
    depth = trial.suggest_int(
        "depth", 2, 12,
        context="网络深度；过深可能导致过拟合和更高延迟",
    )
    return train_and_validate(lr, depth)

study = oa.create_study(
    direction="minimize",
    sampler=oa.AgentSampler(
        backend="mock",
        effort="medium",
        context="小预算图像分类训练",
        history=5,
        explicit_reasoning=True,
        qualitative_notes=True,
    ),
    storage="study.json",
)
study.optimize(objective, n_trials=12)
print(study.best_value, study.best_params)
```

确认流程无误后，把 `backend="mock"` 改为 `claude`、`codex` 或
`opencode`。需要可复现实验时，同时显式设置 `model=` 和 `effort=`。

也可以直接运行 [`examples/quickstart.py`](../../examples/quickstart.py)，或在
[`tutorials/quickstart.ipynb`](../../tutorials/quickstart.ipynb) 中逐步体验。

## Skill 模式

包模式把目标函数视为黑盒；根目录中的 [`SKILL.md`](../../SKILL.md) 让当前编程
智能体先阅读项目代码、理解参数关系，再通过 `study.ask(params)` 和
`study.tell(trial, value)` 驱动同一个 study。可直接从 GitHub 加载到当前编程智能体环境：

```text
$skill-installer install https://github.com/Optim-Agent/optim-agent
```

## 机器学习、量化与推理示例

| 场景 | 示例 | 重点 |
|---|---|---|
| 通用优化 | [`examples/quickstart.py`](../../examples/quickstart.py) | 无额外依赖、离线 mock |
| 机器学习 | [`examples/sklearn_tuning.py`](../../examples/sklearn_tuning.py) | CPU 随机森林、验证集准确率 |
| AI 推理 | [`examples/inference_tuning.py`](../../examples/inference_tuning.py) | 质量下限、P95 延迟和请求成本 |

推理示例使用透明的确定性评估器。接入真实系统时，应替换为固定评测集和实际
服务指标，不要只优化一个未经验证的综合分数。

## 优化轨迹

![智能体与 TPE 的 Branin 优化轨迹](../assets/optimization_trajectory.gif)

动画来自仓库中提交的 seed-0 Branin JSON，展示相同 10 次试验预算下的采样点
和当前最优值。它用于解释搜索过程；正式比较应查看多随机种子基准，而不是单个
动画。可运行 `python scripts/render_trajectory.py` 重新生成。

## 基准与复现

### 无上下文优化数学函数：Branin-2D 与 Ackley-5D

困难函数智能体**不接收任务上下文**：只有通用参数名 `x1...x5`、数值边界和试验历史。所有智能体都使用 medium effort、10 次试验和 5 个随机种子；Random 和 TPE 是不变基线。

#### 顶级智能体

![No-context top-tier hard-function benchmark](../assets/hard_benchmarks_tier.png)

| 方法 | Branin 平均最优值 ↓ | Ackley-5D 平均最优值 ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| GPT-5.5 | 1.326 | 3.960 |
| **Opus-4.8** | **0.398** | **0.061** |
| Sonnet-5 | 3.850 | 0.143 |
| GLM-5.2 | 3.609 | 15.023 |

#### OpenCode 免费智能体

![No-context free-model hard-function benchmark](../assets/hard_benchmarks_free.png)

| 方法 | Branin 平均最优值 ↓ | Ackley-5D 平均最优值 ↓ |
|---|---:|---:|
| Random | 5.008 | 19.639 |
| TPE | 11.395 | 18.843 |
| Big-pickle | 4.734 | 15.951 |
| DeepSeek-V4-Flash | 4.410 | **4.608** |
| Nemotron-3-Ultra | 16.051 | 18.459 |
| **MiMo-v2.5** | **3.682** | 15.597 |

### 调优基于 ResNet 的图像分类器：MNIST 与 CIFAR-10

![MNIST 和 CIFAR-10 五随机种子基准](../assets/classification_benchmarks.png)

Random、Optuna TPE、**GPT-5.5 w/ context** 和 **GPT-5.5 w/o context** 在 5 个随机种子（`0..4`）和 10 次试验下比较。两个 GPT-5.5 条件都固定 `gpt-5.5` 和 medium reasoning effort（`model_reasoning_effort=medium`）；只有 **GPT-5.5 w/ context** 接收 study 目标和 16 个参数的自然语言说明。

| 方法 | MNIST 累计错误 ↓ | MNIST 最终错误 ↓ | CIFAR-10 累计错误 ↓ | CIFAR-10 最终错误 ↓ |
|---|---:|---:|---:|---:|
| Random | 9.174 | 0.648% | 278.920 | 25.072% |
| TPE | 7.166 | 0.580% | 279.936 | 25.596% |
| **GPT-5.5 w/ context** | **5.668** | **0.506%** | **220.994** | **21.322%** |
| GPT-5.5 w/o context | 8.910 | 0.632% | 281.466 | 25.960% |

### 调优梯度提升分类器：信用违约概率

![Five-seed CPU-only GPT-5.5 context benchmark for UCI credit-default HGB tuning](../assets/credit_card.png)

该 CPU-only 基准在 UCI **Default of Credit Card Clients** 数据集（30,000 行、23 个特征、CC BY 4.0）上调优 `HistGradientBoostingClassifier` 的 8 个训练参数。所有方法使用相同数据切分、20 次试验和随机种子 `0..4`。两个 GPT-5.5 条件都使用 high modeling effort、20 条试验历史、显式推理和定性笔记。

| 方法 | 最终验证 log loss ↓ | 留出测试 log loss ↓ |
|---|---:|---:|
| Random | 0.433 | 0.425 |
| TPE | 0.430 | 0.422 |
| **GPT-5.5 w/ context** | **0.428** | **0.422** |
| GPT-5.5 w/o context | 0.433 | 0.427 |

相比匹配的 no-context 对照，上下文将最终验证 log loss 降低 1.13%，将测试 log loss 降低 1.23%。GPT-5.5 在两个指标上都优于 Random 和 TPE。由于配置选择同时使用了验证和测试 loss，测试结果是基准比较，而不是未使用数据上的泛化估计。这是方法学基准，不是生产信用决策系统。

## 并发与持久化

- `.json` 存储适合单写者、可恢复的本地实验。
- `.db`、`.sqlite` 或 `.sqlite3` 使用 SQLite WAL，可由多个进程共享。
- `max_concurrency` 在单个进程中并发执行目标函数；智能体提案仍会串行生成，
  以便读取最新历史。

## 当前限制

- 当前只支持单目标优化；多目标场景需要先显式定义标量效用或约束惩罚。
- 对极便宜、需要成千上万次试验的目标，TPE、GP 或进化算法通常更合适。
- 智能体提案具有随机性；请固定基准种子并保存完整 study。
- 包模式不会读取项目源码，只发送参数、上下文和历史。Skill 模式会按设计读取
  当前项目，因此应独立审查数据和权限边界。

## 故障排查

- **OpenCode 与分布式 study**：OpenCode 当前不支持 optim-agent 的
  `distributed computing` 流程；请使用单进程流程，或为分布式运行选择其他后端。

## 贡献

欢迎提交问题和小型、可验证的 Pull Request。开始前请阅读：

- [`CONTRIBUTING.md`](../../CONTRIBUTING.md)
- [`SECURITY.md`](../../SECURITY.md)
- [`CODE_OF_CONDUCT.md`](../../CODE_OF_CONDUCT.md)
- [`ROADMAP.md`](../../ROADMAP.md)

英文 [`README.md`](../../README.md) 是版本号、基准数值和后端名单的权威来源；
中文文档优先覆盖稳定接口和用户路径，以降低多语言内容漂移。

## 致谢

- 感谢 [Optuna](https://github.com/optuna/optuna) 推广 Study/Trial 接口、提供示例与
  基准中使用的 TPE 基线，并为实用优化工具树立了高标准。
- 感谢 [OpenCode](https://github.com/sst/opencode) 提供本项目在困难函数基准中
  测试的免费模型。

## 许可证

[MIT](../../LICENSE)
