# 材料对应与实现约定

## 你提供的两个参考

- [AIInfraGuide：Transformer 全貌及代码实现](https://caomaolufei.github.io/AIInfraGuide/guides/模块一-前置知识/transformer/32-transformer全貌及代码实现/)：作为架构与主题导航。第 09 关覆盖 Encoder–Decoder 和三种注意力；本文练习代码与讲解独立编写，不复制网页实现。
- [Transformer 计算解剖室](../docs/Transformer_计算解剖室.html)：已随仓库收录，可下载后用浏览器打开。主线对应查表、正弦位置、Pre-Norm、因果多头注意力、GELU FFN、最终归一化、词表投影和 loss。HTML 按原文件复制收录，未修改其内容。

两份材料不是同一个模型。本工作册有意统一使用 Pre-Norm 和 GELU；第 09 关保留 Encoder–Decoder 拓扑，但不宣称逐项复现 2017 原始 Post-Norm/ReLU 配置。默认没有 embedding 的 sqrt(D) 缩放，亦不与 HTML 对齐具体权重或 token 表，因此不要求数值与 HTML 完全相同。HTML 的训练演示仅更新输出头，本工作册要求全参数训练。

## 必须纠正的 mask 语义

网页 9.1 对 `scaled_dot_product_attention` 的 bool mask 解释存在错误。官方 SDPA 中 **True 表示允许参与注意力**；本工作册用 `masked_fill`，约定 **True 表示禁止**。两者不可直接互换。因此检查程序调用官方 SDPA 对照时会传入 `~blocked`。

来源：[PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)。本练习不调用 SDPA 实现学生代码。

## 官方参考（卡在公式/API 时查阅）

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：原始 Transformer、缩放点积注意力和正弦位置公式。
- [PyTorch LayerNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html)：末轴归一化和总体方差。
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)：logits、类别标签与 ignore_index。

补充概念：没有位置编码的 self-attention 对位置置换具有等变性——重排输入会相应重排输出，而不是输出矩阵完全不变。位置编码用于提供显式顺序信息。
