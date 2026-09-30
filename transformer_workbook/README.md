# 自己填完一遍 Transformer

这是一套为你当前进度准备的代码工作册。核心计算留作 TODO，模块声明、离线数据、命令入口和检查程序已经提供。不是完整答案合集。2026-09-30 已将填写过的实现恢复为 20 处 TODO，保留模块骨架、输入检查、提示和验收程序；旧的独立 lesson 练习脚本已移除。

目标：先实现能预测下一个 token 的小型 Decoder-only Transformer，再复用组件实现 Encoder–Decoder。你不必先完成原计划全部阶段，也不代表原计划的优化器、MNIST 等任务已完成。

## 从这里开始

在项目根目录运行，不要进入本目录直接执行某个模块：

```bash
cd models-learning
uv run python -m transformer_workbook.check 01
```

初次会出现 `[待填写] TODO 01a`，退出码 2。这是预期反馈，不是环境坏了。只需使用已有 torch，无新增依赖、下载或环境配置步骤。

1. 先读 [START_HERE.md](START_HERE.md)，认识 `nn.Module` 等新语法。
2. 打开 [embeddings.py](embeddings.py)，只做 TODO 01a、01b。
3. 卡住先看文件内提示，再读 [HINTS.md](HINTS.md) 对应编号。
4. 重新运行本关检查；报错加 `--trace`。将你的代码、输出和解释发给老师。
5. 一关通过后再做下一关，不要一口气填写全部文件。

保留函数名、参数、返回类型及模块属性名；用实现替换 `raise NotImplementedError(...)`。未完成关卡会主动停止，不会用随机值冒充正确结果。构造器内写好的层可直接使用；前向计算由你完成。测试通过不等于已经理解，请填写 [PROGRESS.md](PROGRESS.md) 的解释题。

## 九关路线

| 关卡 | 你填写的文件 | 完成内容 | 检查命令末尾 |
|---|---|---|---|
| 01 | embeddings.py | 位置表、词向量查表、广播相加 | `check 01` |
| 02 | normalization.py | 手写 LayerNorm 与共享仿射参数 | `check 02` |
| 03 | masks.py | 因果、padding、组合 mask | `check 03` |
| 04 | attention.py | 缩放点积、多头拆合、输出投影、支持交叉注意力 | `check 04` |
| 05 | blocks.py | FFN、两次 Pre-Norm 与残差 | `check 05` |
| 06 | model.py | 堆叠 block、最终归一化、词表 logits | `check 06` |
| 07 | objective.py | 错位标签、忽略 PAD 的平均交叉熵 | `check 07` |
| 08 | training.py、generation.py | 训练一步、自回归贪心生成 | `check 08` |
| 09 | seq2seq.py | Encoder、cross-attention、Encoder–Decoder | `check 09` |

完整命令前缀为 `uv run python -m transformer_workbook.`，例如：

```bash
uv run python -m transformer_workbook.check 04 --trace
uv run python -m transformer_workbook.check all
```

01—08 完成主线；09 覆盖参考网页中的第二种架构。单关检查可能调用已完成的前置模块。不要调用 `nn.Transformer`、`nn.MultiheadAttention`、SDPA 或 `nn.LayerNorm` 绕过相应填空；验收程序使用官方组件作为独立数值对照是有意的。允许使用 `nn.Linear`、`nn.Embedding`、`F.gelu`、`F.cross_entropy` 和 autograd。

## 统一的结构约定

```text
ids [B,T]
  → Token Embedding + 正弦位置编码 [B,T,D]
  → 重复 N 次：
      u = LN1(x)
      r = x + MultiHeadAttention(u,u,u,causal+padding mask)
      x = r + FFN(LN2(r))
  → Final LayerNorm
  → Linear(D,V)
  → logits [B,T,V]
  → 与已错位的 targets [B,T] 计算交叉熵
```

- 默认 V=12、D=16、H=4、Dh=4、FFN=32、2 层，CPU 即可。
- 正弦位置编码；Pre-Norm；GELU(tanh)；无 dropout、无 KV cache、无权重绑定。
- 注意力投影和词表投影无 bias；FFN 与 LayerNorm 有共享 bias。
- 不实现 BPE、RoPE、RMSNorm、GQA、FlashAttention。这些不影响本次基础结构练习。
- 同一 batch 只支持右 padding；decoder 的第一个 token 必须有效。PAD=0，BOS=1，EOS=2。
- 注意力 mask 的 `True` **始终表示禁止**；先 mask 再 softmax。每行至少保留一个 key。
- padding key mask 防止读取占位内容；`ignore_index=0` 防止 PAD 标签计入 loss，两者职责不同。
- block 在前向中不 softmax 词表 logits，交叉熵直接接收 logits。

Encoder–Decoder 的 Encoder 不使用 causal mask；Decoder self-attention 用 causal+target padding；cross-attention 的 Q 来自 Decoder，K/V 来自 Encoder，屏蔽 source padding。`S` 与 `T` 不必相等。

## 训练与生成

完成 01—08 后：

```bash
uv run python -m transformer_workbook.run lm --steps 180
```

完成 09 后：

```bash
uv run python -m transformer_workbook.run seq2seq --steps 180
```

观察更新前 loss、最终训练 loss、训练位置准确率和生成文本。主线是四条短句的记忆实验，seq2seq 是两条序列的复制实验。第一种实验某些前缀有多个合理后续，准确率不必达到 100%。loss 应明显低于初始值且无 NaN；不要求每步单调下降，不要求固定生成句子。训练准确率和玩具复制成功都不是泛化能力证明。

本工作册用 SGD，与你此前学过的更新公式一致。`run.py` 不保存权重，每次运行都重新初始化；不会覆盖你的练习。暂不添加 checkpoint 管理以免分散重点。

## 文件分工

- `config.py`、`data.py`、`run.py`：已提供的辅助文件。
- `check.py`：数值、梯度、mask、未来信息隔离等检查；不是待填文件。
- 10 个核心 Python 文件：你动手填写的计算模块；`__init__.py` 是包标记，无需修改。
- [HINTS.md](HINTS.md)：按 TODO 编号分级提示，包含局部 API 示例。
- [PROGRESS.md](PROGRESS.md)：通关和解释记录，初始全部未完成。
- [REFERENCES.md](REFERENCES.md)：材料对应、结构差异与 mask 纠错。

## 完成标准

- 所有关卡检查通过，并能解释每个轴对应什么。
- 修改未来 token 不改变前面位置的预测；追加右 PAD 不改变有效位置预测。
- 小数据 loss 明显下降，能自行逐个生成 token，到 EOS 停止。
- 能解释训练使用真实前缀（teacher forcing），生成时使用自己刚生成的 token。
- 能说明 decoder-only 与 Encoder–Decoder 多出的 cross-attention 差异。
- 最后不看旧代码独立重写一个 Block；这项自测不由脚本自动判定。
