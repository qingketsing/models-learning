# 分关提示：先理解，再看局部写法

每关的一级提示解释关系；二级提示给步骤/API。不必提前阅读后面全部内容。全部函数的输入输出契约见 Python docstring。

## 01：Embedding 与正弦位置

一级：同一个 token 在不同位置查到同一向量，因此需要额外加入位置。位置编码不是训练参数。输入编号不能乘以一个矩阵来代替查表。

二级：先创建 `[L,D]` 零矩阵。对于每个位置 p 和每一对列 2i、2i+1，用相同角度 `p/10000**(2i/D)`，分别填 sin/cos。最后 `unsqueeze(0)` 得到 `[1,L,D]`。

局部广播示例：

```python
position = torch.arange(5, dtype=torch.float32).unsqueeze(1)  # [5,1]
frequency = torch.tensor([1.0, 0.1, 0.01])                  # [3]
angles = position * frequency                              # [5,3]
```

真实频率用 `10000 ** (-torch.arange(0,D,2)/D)`。forward 用 Embedding 查表，截取 `position[:, :T, :]`，相加时广播 batch 轴。初始位置的奇数列等于 1，所以第一个位置并不是完全没有位置信号。

## 02：LayerNorm

一级：每个 token 独立处理自己的 D 个特征，不跨 token 和 batch。

二级：依次计算均值、平方差平均、加 eps 开根号、归一化、乘共享 weight 加共享 bias。不要用无参数的默认 `var()`，其修正项与本题总体方差不同。

局部 API：`x.mean(dim=-1, keepdim=True)`；方差可用 `x.var(dim=-1, correction=0, keepdim=True)`。weight/bias 为 `[D]` 自动广播。保留图，不要用 `torch.tensor(中间结果)` 重新包装而切断梯度。

## 03：Mask

一级：score 的一行是一个 query 对全部 key 的分数。未来 key 和 PAD key 都应屏蔽。

二级：causal 用 `torch.triu`，`diagonal=1` 保留严格上三角。padding 先对 token ID 做等值比较，再用 `[:,None,None,:]` 插入 head/query 轴。两个 bool mask 用 `|` 合并。

T=3 时因果 mask 应是：

```text
False True  True
False False True
False False False
```

不能屏蔽对角线，否则第一个位置无可见 key。不要给 PAD query 整行填 True，这会制造全 -inf 行。只忽略它的标签 loss。输入必须右 padding，这里不扩展到左 padding。

## 04：多头注意力

一级：先 Q/K/V 三次独立投影；每头得到 Dh 个特征；每头独立沿 key 求权重；最后拼接并做输出投影。

二级步骤：

1. 拆头：先 `[B,T,H,Dh]`，再交换 T/H，得到 `[B,H,T,Dh]`。
2. 分数：`q @ k.transpose(-2,-1)` 得到 `[B,H,Tq,Tk]`。
3. 除以 `sqrt(Dh)`，使点积尺度较稳定。
4. 有 mask 时先检查它是否存在全屏蔽行，再 `masked_fill(blocked,float('-inf'))`。
5. `softmax(dim=-1)` 得到每个 query 对 key 的分布。
6. 权重乘 V，得到 `[B,H,Tq,Dh]`。
7. 换回 T/H，再 reshape `[B,Tq,D]`，应用 out_proj。

不要将 transpose 换成 reshape；形状一样不代表元素相同。不要对 Q/K/V 都使用 query 的长度。forward 要返回 `(output, weights)`；调用方加残差时取前者。

局部独立例子：当 q/k 都为零、没有 mask，两个 key 的权重是 `[0.5,0.5]`，结果是两个 V 的平均。可以手工验证。

## 05：FFN 与残差

一级：注意力让不同 token 交换信息；FFN 是每个 token 自己的非线性计算。两个子层都必须回到 D 维，才能与原输入相加。

二级：FFN 依次调用 up、GELU、down。Block 先保存原输入，再 norm1 后 attention；第一次相加产生 r；对 r 做 norm2/FFN，第二次相加也加 r。

Pre-Norm 的形式是 `x + sublayer(norm(x))`，不是 `norm(x + sublayer(x))`。不需要自己写 GELU 导数，autograd 会处理。

## 06：完整语言模型

一级：Block 只输出隐藏表示，还不是词表分数。最终需要 D→V 的投影。

二级：生成同一个 mask 供各层使用；embedding 后用 for 循环遍历 self.blocks，更新隐藏状态；经过 final_norm、lm_head 后返回。禁止每次 forward 新建层，禁止对 logits 做 softmax 后再返回。

先打印每一层 shape；同一位置的向量维度 D 应保持不变，最后投影才变成 V。若删掉 causal mask，仅训练 loss 降得很快并不说明实现正确，很可能在偷看答案。

## 07：标签与 loss

一级：inputs 的位置 t 对应原始 tokens 的 t，target 的位置 t 对应原始 tokens 的 t+1。只错位一次。

二级：`tokens[:, :-1]` 与 `tokens[:, 1:]`。将 logits reshape 为 `[B*T,V]`，targets 为 `[B*T]`。`F.cross_entropy` 接收 logits 和整数标签，用 `ignore_index=pad_id`，默认平均只计有效标签。

切片可能不连续，优先 reshape。EOS 是模型需要预测的真实结束信号，不能与 PAD 一起忽略。没有有效标签时程序应抛错，守卫代码已提供。

## 08：训练与生成

训练一级：仍是你已经写过的四步。区别是参数从整个 model.parameters() 获取。

训练二级：在 train_step 内调用 model.train()、optimizer.zero_grad()；计算 logits 和 token_loss；调用 backward、step；最后返回 loss.item()。当 source 不为 None，用 `model(source, inputs)`，供第 09 关复用。

生成一级：训练时同时预测所有位置；生成时只使用最后位置的预测，然后将新 token 追加到序列。argmax 可以直接作用于 logits，因为 softmax 不改变最大值的位置。

生成二级伪代码：

```text
保存模型原本的 training 标志
切到 eval，复制 prompt
尝试：
    重复最多 max_new_tokens 次：
        若长度到 max_len，停止
        运行模型，提取最后位置分数
        沿词表轴 argmax，保留 [1,1]
        沿 token 轴 cat
        如果追加的是 EOS，停止
    返回完整序列
最终：
    恢复原来的 training 标志
```

局部 API：`logits[:, -1, :]`；`argmax(dim=-1, keepdim=True)`；`torch.cat([a,b], dim=1)`。`@torch.no_grad()` 已提供。eval 改变 Dropout 等层的行为，并不自动关闭梯度；本模型没 dropout，也要养成区分这两件事的习惯。

## 09：Encoder–Decoder

一级：先把源文本变成 memory。Decoder 除了看自己的过去，还需要看 memory。这里源长度 S、目标长度 T 不同也必须运行。

二级：EncoderBlock 复用两次 Pre-Norm 残差，但注意力只屏蔽 source PAD。TranslationDecoderBlock 多一组 cross-attention + norm，因此共三次残差；self-attention 后的结果作为第二步的残差基底，norm2 后作 Q；memory 直接作 K/V。

组装时分别建立 source/target embedding；source_mask 为 `[B,1,1,S]`，可广播到 Encoder `[B,H,S,S]` 与 cross-attention `[B,H,T,S]`。target mask 为 `[B,1,T,T]`。Encoder 循环后做 encoder_norm，Decoder 循环后做 decoder_norm，最后 lm_head。

这是同词表的玩具复制任务；真实翻译可使用不同源/目标词表。主线统一 Pre-Norm，不要照搬网页 Post-Norm 顺序混用。

## 报错先查这里

| 现象 | 优先检查 |
|---|---|
| 梯度全是 None | forward 是否在 no_grad；是否 detach；是否把 Tensor 重新包装 |
| loss 是 NaN | mask 是否全屏蔽；标签是否全 PAD；学习率是否过大 |
| 交叉熵说标签应为 float | logits/target shape 是否相同；是否丢了类别轴 |
| reshape 数量不匹配 | D 是否等于 H×Dh；是否混淆 Tq/Tk |
| 输出每行和为 1 但数值检查失败 | 是否提前给最终 logits 做了 softmax |
| shape 正确但因果检查失败 | mask 方向、对角线、softmax 前后的应用顺序 |
| train loss 下降但生成很差 | 是否重复错位；生成是否取最后位置；是否偷看未来；数据是否太少 |
| bias 变成 [B,T,D] | 应注册 [D] 或 [F]，让广播处理样本和位置 |
