# Models Learning

通过手算、PyTorch 实验和逐步反馈，学习模型开发所需的数学与工程基础。当前重点是 Tensor shape、自动求导、分类训练与优化器。

## 环境与运行

- Python 3.12（见 `.python-version`）
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- PyTorch 与 torchvision：项目已配置官方 CPU wheel 源，当前练习不要求 GPU。

克隆项目后，在 PowerShell 中执行：

```powershell
git clone git@github.com:qingketsing/models-learning.git
cd models-learning
uv sync --locked
uv run python lesson_01.py
```

`uv sync --locked` 按 `uv.lock` 安装依赖并创建 `.venv`；缺少匹配的 Python 时，uv 默认可以自动下载。使用 `uv run` 无需手动激活虚拟环境。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `learning-plan.yaml` | 学习范围、分阶段任务、提示、验收要求和进度记录 |
| `lesson_01.py` | 简化多头注意力的综合练习 |
| `pyproject.toml` | 项目依赖和 PyTorch CPU 下载源 |
| `uv.lock` | 可复现安装所需的依赖锁文件 |
| `.python-version` | 项目使用的 Python 版本 |

虚拟环境、下载的数据集、模型权重和实验输出不提交到 Git。

## 学习路线

计划用 5～7 天建立基础，实际按掌握程度推进：讲一个概念，完成计算或代码，核对结果，再进入下一步。

| 阶段 | 内容 | 当前状态 |
| --- | --- | --- |
| 1 | Tensor、shape、索引与图片展开 | 核心内容已练习，部分基础题留待复查 |
| 2 | 矩阵乘法、广播、维度变换与 Q/K/V | 综合实现已验证通过 |
| 3 | 梯度、计算图、链式法则与 backward | 下一阶段 |
| 4 | logits、softmax、交叉熵与分类训练 | 待学习；已在注意力中练习 softmax |
| 5 | SGD、Momentum、Adam、AdamW 与状态内存 | 待学习 |
| 6 | MNIST 与两个隐藏层的 MLP | 待学习 |
| 7 | activation、独立复写与综合验收 | 待学习 |

上表记录截至 2026-09-24 的进度；详细任务与后续更新以 `learning-plan.yaml` 为准。

## 当前实验：简化多头注意力

`lesson_01.py` 使用随机输入和固定随机种子，串联以下计算：

```text
X [B,T,D]
  → 分别乘 Wq、Wk、Wv
  → Q/K/V [B,T,D]
  → reshape 为 [B,T,H,Dh]，再交换 T 和 H
  → Q/K/V [B,H,T,Dh]
  → Q @ K.transpose(-2,-1) / sqrt(Dh)
  → 分数 [B,H,T,T]
  → 沿最后一轴 softmax
  → 注意力权重 @ V
  → 每个头的输出 [B,H,T,Dh]
  → 交换轴并合并头
  → 输出 [B,T,D]
```

当前设置为 `B=2, T=3, D=8, H=2, Dh=4`。运行时应观察到：

```text
投影后的 q: torch.Size([2, 3, 8])
拆头后的 q: torch.Size([2, 2, 3, 4])
scores: torch.Size([2, 2, 3, 3])
head_output: torch.Size([2, 2, 3, 4])
output: torch.Size([2, 3, 8])
加权求和是否一致: True
```

每个 query 的权重之和应接近 1。脚本还用手动加权求和核对了一个 query 的输出。

此实验用于理解 shape 与前向计算，不包含训练、因果 mask、输出投影或完整 Transformer 层。

## 已遇到的问题与检查方法

- 矩阵乘法先检查共享维度：`[M,K] @ [K,N] → [M,N]`，再检查输出轴的含义。
- 每个 token 共用同一组投影权重；增加 token 数不会增加该线性层的参数量。
- `reshape` 与 `transpose` 用途不同，不能只因目标 shape 一样就互换。
- 拆头后保留 batch/head 轴，对 K 只交换最后两个轴。
- 缩放后的分数还不是权重：先 softmax，再用 `weights @ V` 汇总信息。
- 输出与输入 shape 相同，并不代表数值相同。

下一步从 `y = w*x + b` 和平方误差开始，手算梯度并用 `loss.backward()` 验证。
