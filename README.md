# Models Learning

通过手算、矩阵可视化和 PyTorch 填空练习，理解 Transformer 从 token 输入到预测、训练与生成的完整过程。

当前仓库保留学习材料、旧 Transformer 工作册，以及新增的 GPT 到训练工作册。2026-09-30 已清除工作册中填写过的答案，恢复 **9 个关卡、20 处核心 TODO**；模块声明、输入检查、提示、数据和检查程序仍保留。此前的 `lesson_01.py`、`lesson2.py`、`lesson3.py` 独立练习脚本已移除。

## 从零重写 GPT 到真实训练（新增）

从 `tiny_gpt` 文件夹开始的完整项目已整理为 **18 个关卡、106 处 TODO**。仓库根目录是可填写的练习；`reference_impl/` 保留完整参考实现。脚本参数、校验、运行管理和检查程序已经提供，核心计算与数据处理由你填写。

1. [从零到训练的详细学习指南](docs/GPT从零到训练_学习指南.md)：每关的原理、矩阵形状、实现路径、局部示例、常见错误和过关标准。
2. [全部 TODO 索引](docs/GPT_TODO索引.md)：按编号定位文件与函数。
3. [参考实现运行说明](reference_impl/README.md)：核对答案或运行已完成的版本。

```bash
uv sync --locked
uv pip install --python .venv/bin/python -r requirements-data.txt
source .venv/bin/activate
python scripts/check_gpt_workbook.py 01
```

首次显示 `[待填写]` 是预期结果。填完一关就检查对应编号；遇到实际错误加 `--trace`。完整检查：

```bash
python scripts/check_gpt_workbook.py all
# 验证完整参考答案；不代表练习已经完成
python scripts/check_gpt_workbook.py all --reference
```

学习顺序：Embedding → LayerNorm → mask → Attention → MLP/Block → GPT → Dataset → 训练/验证 → 累积/裁剪 → checkpoint → 学习率 → 字符 tokenizer → device → BPE → 二进制数据 → 长训练保存 → 生成。

本地学习在 CPU 即可；真实数据、CUDA 配置和长训练见学习指南后半部分。GPU 节点使用已有 CUDA PyTorch 并补装 `requirements-data.txt`，不要把仓库的 CPU 源安装命令直接套到节点。历史操作记录也保留在 [数据处理文档](docs/tinystories-training.md) 和 [长训练文档](docs/tinystories-long-training.md)；其中“已生成”“已执行”描述历史节点状态，不表示新克隆包含数据或权重。

## 先看可视化

打开 [Transformer 计算解剖室](docs/Transformer_计算解剖室.html)，沿同一组 token 查看 Embedding、位置编码、归一化、注意力、残差、FFN、logits 和损失的计算。可以点击矩阵元素查看计算来源。

这是单文件 HTML：克隆仓库后可直接用浏览器打开，不需要启动后端或安装 Python。GitHub 文件页通常显示源码，需要下载该 HTML 或克隆后本地打开；本仓库没有配置 GitHub Pages。

macOS 可在仓库根目录运行：

```bash
open "docs/Transformer_计算解剖室.html"
```

HTML 是小型教学演示，采用正弦位置编码和 Pre-Norm Decoder-only 结构，训练按钮只更新输出头；工作册则练习全参数训练，两者不要求数值相同。

## 开始填写工作册

环境使用 Python 3.12 与 [uv](https://docs.astral.sh/uv/)。依赖已在 `pyproject.toml` 和 `uv.lock` 中定义，练习可以在 CPU 上运行。

```bash
git clone --branch main git@github.com:qingketsing/models-learning.git
cd models-learning
uv sync --locked
uv run python -m transformer_workbook.check 01
```

首次出现 `[待填写] TODO 01a` 是预期结果：练习尚未填写，不是安装失败。使用 `uv run` 无需手动激活虚拟环境。

建议按以下顺序阅读：

1. [开始前的新语法说明](transformer_workbook/START_HERE.md)：Module、Linear、Embedding、Parameter。
2. [工作册路线与运行说明](transformer_workbook/README.md)：关卡顺序、接口约定和完成标准。
3. 对应 Python 文件的 TODO：每次只填写当前小步。
4. [分关提示与常见错误](transformer_workbook/HINTS.md)：遇到困难再查看实现路径和局部示例。

## 九个关卡

| 关卡 | 文件 | 任务 |
|---|---|---|
| 01 | `embeddings.py` | token 查表、正弦位置编码 |
| 02 | `normalization.py` | 手写 LayerNorm |
| 03 | `masks.py` | 因果 mask、PAD mask、组合规则 |
| 04 | `attention.py` | Q/K/V 投影、拆头、注意力、合并与输出投影 |
| 05 | `blocks.py` | FFN、Pre-Norm 与残差连接 |
| 06 | `model.py` | 组装 Decoder-only 语言模型 |
| 07 | `objective.py` | 错位标签、忽略 PAD 的交叉熵 |
| 08 | `training.py`、`generation.py` | 参数更新、逐 token 生成 |
| 09 | `seq2seq.py` | Encoder、交叉注意力与 Encoder–Decoder |

表中的 Python 文件均位于 `transformer_workbook/`。检查程序、配置、玩具数据和运行入口已经提供；不需要删除它们或重新编写。

## 检查与训练

填写一关后单独检查，出错时用 `--trace` 查看具体位置：

```bash
uv run python -m transformer_workbook.check 04 --trace
```

完成全部 TODO 后：

```bash
uv run python -m transformer_workbook.check all
uv run python -m transformer_workbook.run lm --steps 180
uv run python -m transformer_workbook.run seq2seq --steps 180
```

检查包含数值对照、梯度、mask、未来信息隔离等行为。训练实验使用离线玩具数据：语言模型学习几条短句，Encoder–Decoder 学习复制序列；这些结果不代表通用语言或翻译能力。

## 资料与记录

- [learning-plan.yaml](learning-plan.yaml)：原始基础学习计划及历史进度，保留供复习；其历史状态不是本轮复写的实时验收状态。
- [工作册进度记录](transformer_workbook/PROGRESS.md)：本轮重新实现时填写。
- [参考与实现约定](transformer_workbook/REFERENCES.md)：参考教程、官方资料、架构差异与 mask 语义说明。

虚拟环境、下载的数据、权重和训练输出不提交到 Git。此次清理只调整当前文件，不重写已有 Git 历史。
