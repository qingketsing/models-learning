# 从创建 tiny_gpt 到训练出模型：18 关重写指南

这份指南复现我们已经完成的项目：从空文件实现一个 GPT 类 Decoder-only 模型，写出训练系统，再接入 TinyStories，最后保存、评估和生成。它不要求你一次理解所有代码。每次只完成一关，运行检查，解释结果，再进入下一关。

我们的实现不是原版 GPT-2 的逐项复现，也没有加载 GPT-2 权重。它使用可学习位置向量、Pre-Norm、因果多头注意力和 GELU MLP；无 dropout、padding、KV cache，输入与输出词表权重不共享。第一版等长序列，所有目标都参与交叉熵。先掌握这套结构，再增加工程优化。

## 0. 从哪里开始，哪些文件需要填写

仓库根目录的 `tiny_gpt/`、`training/`、`data_pipeline/` 和两个指定脚本是练习版。`todo("编号：提示")` 会主动抛出 NotImplementedError；把它换成实际表达式或代码段。不要把 todo 改成 None 或省略号来绕过检查，不要修改检查程序让它通过。

例如原来是：

```python
mean = todo("02.03：沿最后一轴求均值")
```

你需要填写 mean 的计算。这个占位符不是 PyTorch API，只是为了准确指出当前缺少哪一步。

`reference_impl/` 保存本次改成练习前的完整实现。遇到困难先读本指南和局部提示，仍无法完成时再对照同名文件。复制答案不等于理解：看完后关掉参考文件，自己写一遍，再用一个小矩阵解释为什么这样写。

配置、输入检查、循环框架、参数解析、文件校验和后台监护等基础设施多数已提供。`scripts/train.py`、`scripts/train_tinystories.py`、`scripts/launch_tinystories.py` 等入口负责串联你填写的模块。你需要知道它们调用了谁，但不必把全部 argparse 和进程管理重新写一遍。

完整文件布局：

```text
Models-learning/
  tiny_gpt/                 # 01—06：模型核心
  training/                 # 07—12、14、16—17：训练与存储
  data_pipeline/            # 13、15：文本与 tokenizer
  scripts/
    check_gpt_workbook.py    # 18 关检查，不需要改
    train.py                # 离线玩具训练入口，已提供
    train_text.py           # 字符文本入口，已提供
    train_bpe.py             # BPE 命令入口，已提供
    prepare_tinystories.py   # 16：填写 encode_split
    train_tinystories.py     # 真实数据训练入口，已提供
    launch_tinystories.py    # Linux 后台启动入口，已提供
    generate_tinystories.py  # 18：填写加载与生成核心
    evaluate_tinystories.py  # 评估入口，已提供
  examples/corpus/          # 离线小文本
  workbook_support.py      # TODO 占位符工具
  reference_impl/          # 完整参考，目录内可独立运行
  docs/GPT_TODO索引.md       # 全部 TODO 的文件、函数、任务
```

## 0.1 配置本地学习环境

在仓库根目录运行：

```bash
uv sync --locked
uv pip install --python .venv/bin/python -r requirements-data.txt
source .venv/bin/activate
python scripts/check_gpt_workbook.py 01
```

之后统一使用这个环境中的 `python`。第二条命令安装 BPE 与二进制数据依赖；这些依赖通过 requirements-data.txt 单独管理。再次执行 uv sync 可能移除额外依赖，需要重新安装它们。不要在本指南中混用会自动同步环境的 uv run 命令。

预期第一次输出 `[待填写] 01: TODO 01.01...`，退出码 2 表示还没填写。真正错误显示 `[未通过]`，退出码 1；该关全部检查通过则退出码 0。

```bash
python scripts/check_gpt_workbook.py 04 --trace
python scripts/check_gpt_workbook.py all
```

`--trace` 用于定位实际错误。检查某关时仍提示前面一关的 TODO，说明依赖还没有填完。例如训练检查先创建 GPT，模型未完成时无法验证训练。

不修改任何练习也能检查参考实现：

```bash
python scripts/check_gpt_workbook.py all --reference
```

完整答案全部通过不意味着你的练习通过；默认不加 --reference 才是在检查你填写的目录。

## 0.2 用三个符号贯穿全项目

- B：一次前向计算的序列数量（batch size）。
- T：每条序列的 token 数（序列长度）。
- D：每个 token 的特征数（d_model）。
- V：词表大小。H：注意力头数。Dh=D/H：每个头的特征数。

贯穿练习的小例子：B=2、T=3、D=8、H=2、Dh=4、V=11。

```text
ids                  [2,3]       long
embedding            [2,3,8]     float
q/k/v（拆头前）      [2,3,8]
q/k/v（拆头后）      [2,2,3,4]
scores/weights       [2,2,3,3]
context              [2,2,3,4]
合头、Block 输出     [2,3,8]
logits               [2,3,11]
flatten logits       [6,11]
flatten targets      [6]
loss                 []          标量 Tensor
```

学习动作：在纸上写出每次变换的形状，再写代码。不要把 V、T、D 混为一谈：一个是类别数，一个是位置数，一个是特征数。

## 01. Embedding：编号如何变成向量

文件：`tiny_gpt/embeddings.py`。前置：理解 Module、Embedding 与 Tensor 的区别。

任务：完成 token 表、位置表、查表、生成位置编号和相加的 6 个 TODO。

1. 创建 token 表，参数形状 [V,D]。调用 nn.Embedding(V,D)，不是手动创建 [B,T,D] 参数。
2. 创建位置表，参数形状 [max_len,D]。它和 token 表一样会被训练。
3. 将 ids 输入 token 模块，得到 [B,T,D]。
4. 生成 0 到 T-1 的位置编号，形状 [T]，设备跟 ids 一致。
5. 查位置表得到 [T,D]，在最前面补一个轴得到 [1,T,D]。
6. 两者相加。长度为 1 的 batch 轴自动广播到 B。

查表示例（D=2，只为方便看）：

```text
token表第1行=[10,20]，第2行=[30,40]
位置0=[1,2]，位置1=[3,4]
输入 [1,2]
输出 [[11,22],[33,44]]
```

不要把 token 编号当作数值特征直接乘 W。编号 2 并不表示“语义是编号 1 的两倍”。Embedding 是按行查参数表。

局部语法示例：`torch.arange(3)` 产生 `[0,1,2]`；`tensor.unsqueeze(0)` 在最前面加一个轴。D 是由模型配置确定的，不是词长。

检查：`python scripts/check_gpt_workbook.py 01`。它验证数值相加、形状、两个表都能获得梯度，以及超过 max_len 的输入被拒绝。

过关后自问：为什么所有 batch 共用同一张位置表？为什么同一个 token 出现在不同位置时输出不同？

## 02. LayerNorm：对每个 token 的特征归一化

文件：`tiny_gpt/normalization.py`。

每个 token 独立沿 D 计算：

```text
mean = sum(x_d)/D
var  = sum((x_d-mean)^2)/D
z    = (x-mean)/sqrt(var+eps)
out  = z*weight+bias
```

例子：一个 token 是 [1,3]，均值=2，总体方差=1。不考虑极小的 eps，归一化得到 [-1,1]。另一个 token [10,14] 使用自己的均值=12、方差=4，不与前一个 token 混在一起算。

步骤：

1. weight 初始化全 1、bias 全 0，形状都为 [D]，必须包装成 nn.Parameter。
2. 求 mean，dim=-1、keepdim=True，形状 [B,T,1]。
3. 求总体方差，correction=0、keepdim=True；也可以平方差求平均。
4. 标准差是 sqrt(var+eps)，不是 var，也不是 sqrt(var)+eps。
5. 归一化后逐元素乘 weight 加 bias，结果仍 [B,T,D]。

为什么保留轴？[B,T,1] 可以明确广播到 [B,T,D]。如果方差被压成 [B,T]，广播可能报错，也可能在尺寸碰巧相同时悄悄沿错误方向运算。

检查：`python scripts/check_gpt_workbook.py 02`。对照 PyTorch LayerNorm，检查参数注册、常量输入无 NaN 和输入梯度。本关禁止用 nn.LayerNorm 替代计算。

## 03. 因果 Mask：当前位置不能偷看后文

文件：`tiny_gpt/masks.py`。全项目约定：bool 中 True 表示禁止关注。

T=3：

```text
         key0  key1  key2
query0   False True  True
query1   False False True
query2   False False False
```

步骤：创建 bool [T,T] 的全 1 张量；保留 diagonal=1 的严格上三角；补两个前导轴得到 [1,1,T,T]。当前框架的 triu 调用已经提供，你只填标出的占位符。

数学条件：第 i 行、第 j 列在 j>i 时屏蔽。主对角线允许看自己，不能设置 diagonal=0。

前两个轴长为 1，后续广播到每个 batch 和每个 head。这里没有 padding mask：第一版用固定长度连续 token 块。

检查：`python scripts/check_gpt_workbook.py 03`。

## 04. 多头 Attention：从谁那里取信息

文件：`tiny_gpt/attention.py`。线性层声明已提供，填写拆头、合头与前向的 TODO。

Q 是当前 token 的查询特征，K 是其他位置用于匹配的特征，V 是匹配后要汇总的内容特征。它们都由同一个 x 经不同的可训练 Linear 得到，不是你手工标注的关键词。

先完成 split_heads 和 merge_heads：

```text
[B,T,D] → reshape [B,T,H,Dh] → transpose [B,H,T,Dh]
[B,H,T,Dh] → transpose [B,T,H,Dh] → reshape [B,T,D]
```

不能直接把 [B,T,D] reshape 成 [B,H,T,Dh]：元素虽然没丢，token/head 的对应关系变了。Dh 使用整数除法，不能写 D/H 得到浮点尺寸。

再完成前向：

1. q_proj(x)、k_proj(x)、v_proj(x)，得到三个 [B,T,D]。
2. 分别拆头到 [B,H,T,Dh]。
3. scores=QKᵀ/sqrt(Dh)，得到 [B,H,T,T]。K 只交换最后两个轴。
4. 用 causal mask 把禁止位置填为负无穷。
5. 沿最后的 key 轴做 softmax，每个 query 的权重和为 1。
6. context=weights@V，得到 [B,H,T,Dh]。
7. 合头并调用 out_proj，输出 [B,T,D]。

权重例子：一行权重为 [0.25,0.75]，两个 value 为 [2,0]、[0,4]，汇总结果是 [0.5,3]。这一步是矩阵乘法，不是选出唯一一个 value。

为什么填 -inf？exp(-inf)=0，因此禁止位置的 softmax 权重为 0。填 0 时 exp(0)=1，仍会参与关注。因果 mask 保留自己，所以不存在整行全部屏蔽的问题。

检查：`python scripts/check_gpt_workbook.py 04`。包含拆头逆变换、完整数值对照、概率和、未来权重为零，以及修改未来输入不会改变过去输出。

## 05. MLP 和 Transformer Block：变换特征、保留原信息

文件：`tiny_gpt/mlp.py`、`tiny_gpt/block.py`。

MLP 对每个 token 独立运算，不混合序列位置：

```text
[B,T,D] → Linear(D,d_ff) → GELU → Linear(d_ff,D) → [B,T,D]
```

先扩维、再激活、再降维。局部调用形如 `F.gelu(hidden, approximate="tanh")`。没有非线性时，两层线性变换仍可以合并成一次线性变换。

Block 采用 Pre-Norm：

```text
normalized = LN1(x)
a = Attention(normalized)
r = x + a
normalized = LN2(r)
m = MLP(normalized)
output = r + m
```

两个 LayerNorm 参数独立。第一次残差加回原始 x，第二次加回第一次残差 r。不要把 normalized 当作残差，也不要在 MLP 内再加一次残差。

检查：`python scripts/check_gpt_workbook.py 05`。对照完整运算顺序和 MLP 激活。

## 06. GPT：把模块连接起来

文件：`tiny_gpt/model.py`，配置 `tiny_gpt/config.py` 已提供。

初始化时创建 embedding、独立的 n_layers 个 Block、final_norm、lm_head。ModuleList 用来注册模块，但不是直接调用的层；用 for 遍历它。不能写 `[同一个block]*n_layers`，那会多次引用同一份参数。

前向顺序：

```text
ids → embedding → block0 → block1 → ... → final_norm → lm_head
```

循环里每次使用上一层输出。lm_head 是 Linear(D,V)，默认不共享 Embedding 权重。返回原始 logits，不做 softmax 或 argmax，否则训练损失拿不到完整的分数信息。

logits[b,t,:] 的含义：读到序列 t 位置后，对下一个 token 的 V 个候选分数。

检查：`python scripts/check_gpt_workbook.py 06`。输出形状、独立层、因果性、所有参数梯度。

至此只完成了“模型能计算”，尚未“模型学会了”。随机参数也可以产生 logits，但它们没有训练意义。

## 07. Dataset：制造下一词预测任务

文件：`training/data.py`。

```text
tokens = [1,2,3,4,5,6]，T=3
样本0：x=[1,2,3]，y=[2,3,4]
样本1：x=[2,3,4]，y=[3,4,5]
样本2：x=[3,4,5]，y=[4,5,6]
```

这里步长为 1，样本数 N-T。Dataset 返回单条 [T]；DataLoader 再堆叠成 [B,T]。shuffle 只打乱样本顺序，不把样本中的文字打乱。

任务：完成 len 和两个切片。标签已经在 Dataset 中错位，训练时不能再次移动 y。

检查：`python scripts/check_gpt_workbook.py 07`。包含首尾样本和越界行为。

## 08. train_step：一次参数更新

文件：`training/trainer.py` 中 train_step。

固定顺序：

```text
model.train → zero_grad → model(x) → cross_entropy → backward → step
```

x/y 均为 [B,T]，logits 为 [B,T,V]。展开为 [B*T,V] 与 [B*T]，每一行分数对应同位置的一个目标编号。

交叉熵对一个位置的计算：

```text
p_target = exp(logits_target) / sum(exp(logits_v))
loss_position = -log(p_target)
loss = 所有目标位置的平均
```

例如两类分数相等，正确类别概率为 1/2，loss≈0.6931。正确类别概率 0.9 时 loss≈0.1053。无需自己先 softmax，F.cross_entropy 接收 logits 并内部计算稳定的 log-softmax。

loss.shape 是 []，它是标量 Tensor，仍连接着整个计算图。backward 沿图计算每个可训练参数的梯度。loss.item() 只取出一个 Python 数字用于记录，不能对这个 Python 数字 backward。

检查：`python scripts/check_gpt_workbook.py 08`。检查返回的是更新前损失，并且参数实际改变。

## 09. evaluate：检查模型，而不是继续学习

文件：`training/trainer.py` 中 evaluate。本关需自己补一个函数体。

实现路径：

1. 记住 `was_training = model.training`。
2. 初始化 total_loss=0、total_tokens=0。
3. 用 try/finally 保证不论成功或报错，都能恢复原模式。
4. try 内先 model.eval，再进入 torch.no_grad。
5. 遍历 batch，搬到模型设备，前向计算交叉熵。
6. 每个 batch：total_loss += loss.item()*y.numel()，total_tokens += y.numel()。
7. 如果 token 数为 0 抛 ValueError，否则返回二者之比。
8. finally 中 `model.train(was_training)`。

平均值不能简单再平均。例如两个 batch 分别含 2、3 个目标，平均 loss 分别 1、3；总平均是 (1*2+3*3)/5=2.2，而不是 (1+3)/2=2。

不调用 backward、step 或 zero_grad。eval 切换层行为，no_grad 才关闭梯度记录。

检查：`python scripts/check_gpt_workbook.py 09`。验证加权平均、参数不变、无梯度和空数据报错后仍恢复模式。

## 10. 梯度裁剪和累积：控制更新方式

文件：`training/trainer.py` 的 clip_gradients、train_accumulated_step。

先做梯度裁剪：在 backward 之后、optimizer.step 之前，把所有参数梯度的整体 L2 范数限制在 max_grad_norm。使用 clip_grad_norm_，开启非有限梯度检查。它限制更新所用的梯度，不是把模型参数值裁到 [-1,1]。

再做累积：整组只 zero_grad 一次；每个小 batch 分别 forward/backward；整组最后裁剪、step 一次。当前小 batch 的平均 loss 需乘当前 token 数/整组 token 数。

例子：组内第一批 6 个 token、第二批 3 个，分别乘 6/9 与 3/9。不能固定除以 2，更不能总除以配置累积次数，因为最后一组可能不足。

```text
batch=4，累积4 → 4 次前后向 + 1 次更新
batch=16，累积1 → 1 次前后向 + 1 次更新
```

数学上的有效 batch 相同，实际 GPU 并行程度不同；浮点运算顺序可能产生细微差异。

检查：`python scripts/check_gpt_workbook.py 10`。对比完整 batch 和不等大小累积组的 SGD 更新，并检查裁剪后的范数。

## 11. checkpoint：让训练成果离开内存

文件：`training/checkpoint.py`。

保存的结构已经提供：model、optimizer、epoch、global_step、model_config、train_config，可选 extra_state。

任务：取两个 state_dict；把字典写入 temporary；读取 checkpoint 到 CPU；恢复参数、优化器以及计数。

模型加载顺序：

```text
读取 model_config → 创建同结构模型 → model.to(device)
→ 创建优化器 → 恢复模型 state_dict → 恢复优化器 state_dict
```

只为生成时可以不恢复优化器；继续训练需要 Adam 的动量与计数。权重文件不是 tokenizer，文字与编号映射也要匹配。

同目录临时文件和 os.replace 框架已提供：如果写入失败，旧 checkpoint 不被半截新文件覆盖。请保留 finally 清理逻辑。

检查：`python scripts/check_gpt_workbook.py 11`。检查参数、优化器、计数恢复和配置不匹配被拒绝。第 17 关还会模拟写入失败。

当前实现不保存随机数与迭代器位置，因此恢复后重新打乱，不能保证严格复现不中断的训练。

## 12. 学习率：预热后衰减

文件：`training/schedule.py`。step 从 1 开始，指即将执行的第几次 optimizer.step，不是小 batch 的序号。

分三段：

1. step≤warmup：lr=base_lr*step/warmup_steps。
2. step≥total：lr=min_lr。
3. 中间：progress=(step-warmup)/(total-warmup)，从 base_lr 插值到 min_lr。

例如 base=.01、min=.001、warmup=10、total=150：第1步 .001，第10步 .01，第80步 .0055，第150步 .001。

set_lr 遍历 optimizer.param_groups，修改每个 group['lr']。训练在更新之前设置它。不要重建 optimizer 去修改学习率，那样会丢状态。

检查：`python scripts/check_gpt_workbook.py 12`。

完成 01—12 后，先跑一次离线集成训练：

```bash
python -m scripts.train --device cpu --epochs 2 --checkpoint checkpoints/toy.pt
```

这只证明流程通畅，不证明真实语言能力。下一关负责把文字变成输入编号。

## 13. 字符 Tokenizer：先理解最简单的文字映射

文件：`data_pipeline/tokenizer.py`。vocab_size 属性已提供，填写 6 个函数体。

实现顺序：

1. from_text：检查非空字符串；去重并排序字符；把 <unk> 放在编号 0；用 cls(tokens) 创建实例。
2. __init__：验证词表列表、无重复、0 是 <unk>；复制列表；用 enumerate 建 token_to_id。
3. encode：逐字符查表，没见过的字符返回 0。
4. decode：验证每个编号；按原编号顺序查列表，用 ''.join 连接。
5. save：UTF-8 JSON 保存 type='char'、version=1 和 tokens。
6. load：读取并验证 type/version；直接从 tokens 重建，不重新排序。

```text
训练文字="我 爱我"
词表=["<unk>"," ","我","爱"]
encode("我 爱")=[2,1,3]
encode("猫")=[0]
decode([0])="<unk>"
```

空格、换行、标点都保留。不能用 strip 删除空白，不能用验证集扩词表，不能编码一个新字符时临时给它新增编号。未知字符已经丢失信息，不能指望解码恢复原字。

检查：`python scripts/check_gpt_workbook.py 13`，完整数据检查入口也保留为 `python -m scripts.check_tokenizer`。

## 14. device：参数和数据在哪里计算

文件：`training/device.py`。

resolve_device 的路径：auto 优先 CUDA，再检查 Mac MPS，最后 CPU；显式 cpu 直接使用；显式 cuda 验证 CUDA 可用和设备编号；显式 mps 验证支持；其他类型拒绝。

model_device 从一个参数获取设备。move_batch 只搬当前 x/y，不把整个 Dataset 放 GPU。long 编号保持 long；神经网络参数使用浮点。模型先迁移到设备，再创建 optimizer。

检查：`python scripts/check_gpt_workbook.py 14` 在 CPU 上验证接口。具备相应设备时再使用 `python -m scripts.check_device` 验证跨设备行为。没有 NVIDIA GPU 不应该填写一个假 CUDA 设备去通过检查。

## 15. TinyStories 和 BPE：接入真实故事

文件：`data_pipeline/tinystories.py` 的 iter_stories、train_bpe。sha256 工具已提供。

首先实现故事迭代器，别把整个 2 GB 文本 read_text 进内存：

1. 用 UTF-8 文件逐行读取，暂存当前故事的行。
2. 遇到独占一行的 `<|endoftext|>` 时，合并已有行，清除首尾空白。
3. 非空故事才 yield，空故事不算入 limit；清空行缓存。
4. 文件结束时仍有文本，要返回最后一篇。
5. limit 为 None 时不限制，为正数时只返回前 N 篇，0 不是此函数的合法值。

例子：`A\nB\n<|endoftext|>\nC` → 两篇：`A\nB`、`C`。内部换行保留。

再实现 BPE，调用成熟的 tokenizers 库，不在本关手写 BPE 合并算法：

- 创建 models.BPE 模型和 Tokenizer 容器。
- pre_tokenizer 使用 ByteLevel(add_prefix_space=False)。
- decoder 使用 ByteLevel。
- BpeTrainer 设置词表大小、min_frequency=2、special_tokens=[EOS]。
- initial_alphabet 使用完整 ByteLevel.alphabet()，保证 256 个字节被覆盖。
- 用迭代器喂训练集，统计实际故事数；返回 tokenizer 与 count。
- 词表至少容纳 256 字节和一个 EOS；空训练集报错；检查普通 Unicode 往返。

关键概念：tokenizer 训练与 GPT 训练是两件事。BPE 决定编号体系，GPT 学习这些编号的出现规律。词表一旦固定，后续预处理、模型和生成全部沿用它。保留标记 EOS 不属于普通 Unicode 往返示例。

检查：`python scripts/check_gpt_workbook.py 15`。

真实数据下载地址与版本校验可参考 [数据处理文档](tinystories-training.md)。新机器自行提供数据文件；仓库不包含下载后的大文件。

## 16. 二进制编码和 memmap Dataset

文件：`scripts/prepare_tinystories.py` 的 encode_split，`training/binary_data.py`。

encode_split 分成三段写，不要一次写完整函数：

**A. 输入与编码**

校验 EOS 存在、词表≤65536。创建 destination.bin.tmp，逐批取故事（如 512 篇）。每篇内容不能含非分隔行的保留 EOS。用 encode_batch，关闭额外特殊 token；每篇编码结果后追加恰好一个 eos_id。

**B. 写入与计数**

将当前批的编号拼成列表，转 np.asarray(..., dtype='<u2')，写 bytes。累计 stories、tokens。每个编号占 2 字节，不要用默认 numpy int64 存文件。缓存 vocab_size 一次，不要在每篇循环反复查询。

**C. 校验与发布**

确认非空；文件字节数=2*tokens；用 memmap 分块检查编号范围，EOS 个数=故事数且末尾是 EOS。删除读映射引用后用 replace 把临时文件变成最终文件。返回的字典必须包含 stories、tokens、bytes、sha256、source、source_sha256、requested_limit，才能被已提供的 CLI 写进 metadata.json。

局部写法示例：

```python
# 只是展示格式，不是整个 encode_split 的答案
payload = np.asarray([1, 257, 0], dtype="<u2").tobytes()
# 三个编号占六字节
```

主函数最后才生成 metadata.json，把它作为处理完成的标志。拒绝覆盖已有结果；失败后可换新的输出目录重做。

BinaryTokenDataset 的初始化和延迟 memmap 框架已提供。长度=(N-1)//T；样本 i 从 i*T 开始读 T+1 个编号；复制并转 int64，再转 Tensor，返回 chunk[:-1]、chunk[1:]。

这里与第 07 关不同：步长从 1 变成 T，减少大规模训练中的重叠读取。末尾不足完整样本的 token 舍弃。连续故事通过 EOS 拼接，本版不额外隔离跨故事注意力。

检查：`python scripts/check_gpt_workbook.py 16`，会验证首尾样本、EOS 和实际二进制内容。

## 17. latest、best 与历史文件：管理长任务

文件：`training/run_checkpoints.py` 的 save_run_checkpoint。atomic_copy 工具已提供。

实现步骤：

1. 验证 keep≥1，创建目录。
2. 历史文件名使用 step-XXXXXXXX.pt，按步数补零便于排序。
3. 调用第 11 关的 save_checkpoint，保存 val_loss、best_val_loss、best_step 到 extra_state。
4. improved=True 时原子更新 best.pt；否则保持最佳模型不变。
5. 原子更新 latest.pt。
6. 排序所有历史 step 文件，只删除超出最近 keep 个的部分，不删除 best/latest。

best 指固定验证子集 loss 最低，不是人工挑选最好的故事。latest 用于恢复，best 常用于生成测试。新任务使用新目录，恢复任务保持训练配置一致。

检查：`python scripts/check_gpt_workbook.py 17`。含最佳记录、历史保留、模型/Adam 恢复及模拟半截写入失败后旧文件不变。

## 18. 生成：把模型输出接回输入

文件：`scripts/generate_tinystories.py`。参数解析、设备校验、EOS 判断和输出格式已提供。

先完成加载：读取 checkpoint，核对 tokenizer 哈希，用 model_config 重建 GPTConfig 和 GPT，load_state_dict 后迁移设备并 eval。别只创建随机模型就开始生成。生成使用 inference_mode，无需恢复 optimizer。

接下来填写循环：

```text
prompt → tokenizer.encode → long [1,T]
→ model → 取 logits[:, -1, :]，得到 [1,V]
→ 选 next_id [1,1]
→ 若 EOS 则停，否则 cat 到输入最后
→ 重复
```

为什么只取最后位置？那一行表示“读完整个当前输入后，下一个 token 的预测”。其他位置预测的是它们各自的下一个 token。

先实现贪心：argmax(dim=-1,keepdim=True)。再做采样：

1. 除以正的 temperature。较小温度使分布更集中，不会增加模型知识。
2. top_k>0 时 topk 筛选候选，softmax 后 multinomial 抽取候选索引。
3. 用 indices.gather 把候选位置映射回实际词表编号，不要把候选索引直接拼进 ids。
4. top_k=0 时对完整词表采样。
5. 保留最近 max_len 个输入，避免访问越界位置表；这会丢失更早上下文，且位置编号重新从零开始。
6. 用 torch.cat(...,dim=1) 拼新 token，最终 decode 整段编号。

检查：`python scripts/check_gpt_workbook.py 18`。它创建小型测试模型，不下载权重，检查贪心 EOS 终止和采样流程。该检查证明接口可用，不证明生成内容好。

## 19. 全部完成后：从小检查走到正式训练

先跑：

```bash
python scripts/check_gpt_workbook.py all
python -m scripts.train --device cpu --epochs 2 --checkpoint checkpoints/toy-complete.pt
```

参考实现也可单独运行：

```bash
cd reference_impl
../.venv/bin/python -m scripts.train --device cpu --epochs 2 --checkpoint checkpoints/reference-toy.pt
cd ..
```

本地下载数据和训练输出不提交 Git。参考版本独立运行时，输出位于 reference_impl 内；仓库忽略规则也覆盖这些产物。

### GPU 节点环境

把填写完成的代码复制或 git clone 到节点。不能把本地 .venv 复制过去。仓库 pyproject 的 PyTorch 源是 CPU，节点不要照搬 uv sync 安装去覆盖 CUDA PyTorch。

如果节点已经有可用的 CUDA PyTorch：

```bash
# 用节点上确认能 import torch 且 CUDA 可用的 Python 执行
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-data.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

如果没有 CUDA PyTorch，先根据节点驱动和官方安装说明安装适配版本，不把某个历史版本号写成所有机器通用命令。

### 数据准备顺序

使用你自己的原始数据目录、输出目录：

```bash
python -m scripts.train_bpe \
  --train-file /root/autodl-tmp/datasets/TinyStories/TinyStories-train.txt \
  --output-dir /root/autodl-tmp/datasets/TinyStories/bpe8k \
  --limit 10000 --vocab-size 8000

python -m scripts.prepare_tinystories \
  --raw-dir /root/autodl-tmp/datasets/TinyStories \
  --tokenizer /root/autodl-tmp/datasets/TinyStories/bpe8k/tokenizer.json \
  --output-dir /root/autodl-tmp/datasets/TinyStories/bpe8k-full \
  --train-limit 0 --val-limit 0
```

已有 tokenizer/data 目录时不要重复覆盖，换新输出路径或沿用已校验的现有数据。先用 10000/1000 的小子集验证也可以，但要明确这不是全量编码。训练 tokenizer 本身仍只读前 10000 篇训练故事，之后用固定词表编码完整数据。

### 计算一个 epoch 的更新次数

```text
样本块数 = floor((N-1)/T)
小 batch 数 = ceil(样本块数 / batch_size)
更新数 = ceil(小 batch 数 / accumulation_steps)
```

本次历史数据 N=459226619，T=256：1793853 个样本块。

- batch=16、累积1：112116 次更新。
- batch=32、累积1：56058 次更新。

尾 batch 也参与训练。检查你的 metadata 后再计算，不把这两个数当作所有数据集通用值。

从头训练一个 epoch 的历史配置：

```bash
python -m scripts.launch_tinystories \
  --name my-gpt-full-epoch1 \
  --steps 56058 --warmup-steps 1000 \
  --batch-size 32 --accumulation-steps 1 --eval-batch-size 4 \
  --eval-every 1000 --eval-batches 100
```

这是启动真正后台任务的命令，只在填写并验证完成、数据准备完成后执行。项目默认模型 V8000、D256、6层、4头、FFN1024；修改架构前要重新建立训练计划与配置。查看日志：`tail -f runs/my-gpt-full-epoch1/train.log`；Ctrl+C 只停止 tail，不停止后台训练。正常完成应出现对应最终步数和 completed 状态。

恢复时重复原来所有参数并追加 --resume。当前实现重新打乱数据，不能保证中断恢复后的固定步数恰好将每个样本遍历一次。状态文件不是实时心跳，机器重启后 running 状态可能过期。更多操作见 [长训练文档](tinystories-long-training.md)。

### 生成与评估

```bash
python -m scripts.generate_tinystories \
  --checkpoint runs/my-gpt-full-epoch1/best.pt \
  --tokenizer /root/autodl-tmp/datasets/TinyStories/bpe8k-full/tokenizer.json \
  --prompt "Once upon a time" --sample --temperature 0.8 --top-k 40

python -m scripts.evaluate_tinystories \
  --run-dir runs/my-gpt-full-epoch1 \
  --data-dir /root/autodl-tmp/datasets/TinyStories/bpe8k-full \
  --output-dir outputs/my-gpt-evaluation
```

该评估入口目前按长度 256 构造验证块，适用于本教程配置；更换上下文长度时需同时修改评估范围并记录。全验证样本 loss 与固定前 100 batch loss 是不同指标，不要混在同一张对照表里。

## 20. 完成标准和复盘

能够运行不等于掌握。完成后应做到：

- 不看代码能画出 ids→logits→loss 的形状链。
- 能用一个 token 和两个 value 解释 LayerNorm 与 Attention。
- 能解释 causal mask 为什么放在 softmax 前。
- 能解释训练标签、有效 batch、global_step、epoch 的区别。
- 能从 checkpoint 创建同结构模型，并使用同一个 tokenizer 生成。
- 能对相同验证范围、相同 prompt/seed 比较模型。
- 能指出故事生成里的重复、指代错误和因果不连贯，不只看 loss。

本次历史结果：约 890 万参数，完整一遍训练 56058 步，用时约 24 分 26 秒；全验证完整样本 loss 为 1.7804，PPL≈5.93。10k 步模型在同一范围 loss 为 2.3877。结果依赖数据、环境、种子与实现，不是每次练习的强制数值目标。

已完成的是小型语言模型预训练流程，不是通用聊天模型。可继续研究多轮训练、严格恢复、BF16、torch.compile、融合 Attention、KV cache、任务评测；这些尚未作为本轮实现成果。

## 常见问题快速定位

| 现象 | 先检查 |
|---|---|
| integer tensor cannot require gradients | 参数用 float，编号用 long |
| ModuleNotFoundError | 在仓库根目录运行 python -m scripts...，包内用相对导入 |
| shape 错误 | 写出 B/T/D/H/Dh/V，别直接猜 reshape |
| LayerNorm 广播异常 | mean/var 的 keepdim=True |
| mask 后仍看未来 | True 位置填 -inf，softmax 沿最后轴 |
| 因果检查失败 | K 转置轴、mask、模型循环是否使用正确输入 |
| 参数不更新 | nn.Parameter、ModuleList、backward、step 顺序 |
| loss 出现 NaN | 方差 eps、mask 是否整行屏蔽、梯度有限性 |
| 交叉熵 shape 不匹配 | logits [B*T,V]，targets [B*T] |
| 未知字符很多 | 只用训练集建词表；字符 tokenizer 有覆盖限制 |
| checkpoint 加载失败 | model_config、train_config、tokenizer 是否一致 |
| GPU 利用率较低 | 对比 token/s，不只看百分比；小 batch 与同步开销 |
| 每次采样结果相同 | seed 固定；更换 seed 才改变抽样序列 |
| 生成到一半截断 | max_new_tokens 上限，不一定是模型主动结束 |

全部编号与位置见 [GPT_TODO索引](GPT_TODO索引.md)。每关完成后，在 [学习进度表](GPT学习进度.md) 记录本轮进度。
