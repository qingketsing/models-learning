# GPT TODO 索引

共 106 处 TODO。先阅读 [学习指南](GPT从零到训练_学习指南.md)，再按编号完成。编号左边是关卡，右边是该关内的任务号。

| 编号 | 文件与函数 | 实现目标 |
| --- | --- | --- |
| 01.01 | [tiny_gpt/embeddings.py](../tiny_gpt/embeddings.py) · `GPTEmbedding.__init__` | 创建 nn.Embedding(V,D)，它是模块，不是直接参与 @ 的矩阵 |
| 01.02 | [tiny_gpt/embeddings.py](../tiny_gpt/embeddings.py) · `GPTEmbedding.__init__` | 创建 nn.Embedding(max_len,D)，位置参数也参与训练 |
| 01.03 | [tiny_gpt/embeddings.py](../tiny_gpt/embeddings.py) · `GPTEmbedding.forward` | 调用 token 查表，得到 [B,T,D] |
| 01.04 | [tiny_gpt/embeddings.py](../tiny_gpt/embeddings.py) · `GPTEmbedding.forward` | torch.arange(T)，设备与 input_ids 相同 |
| 01.05 | [tiny_gpt/embeddings.py](../tiny_gpt/embeddings.py) · `GPTEmbedding.forward` | 查位置表并补 batch 轴：[T,D]→[1,T,D] |
| 01.06 | [tiny_gpt/embeddings.py](../tiny_gpt/embeddings.py) · `GPTEmbedding.forward` | 相加并广播，返回 [B,T,D] |
| 02.01 | [tiny_gpt/normalization.py](../tiny_gpt/normalization.py) · `LayerNorm.__init__` | 用 nn.Parameter 包装全 1 的 [D] 张量 |
| 02.02 | [tiny_gpt/normalization.py](../tiny_gpt/normalization.py) · `LayerNorm.__init__` | 用 nn.Parameter 包装全 0 的 [D] 张量 |
| 02.03 | [tiny_gpt/normalization.py](../tiny_gpt/normalization.py) · `LayerNorm.forward` | 沿 dim=-1 求平均，keepdim=True |
| 02.04 | [tiny_gpt/normalization.py](../tiny_gpt/normalization.py) · `LayerNorm.forward` | 总体方差 correction=0，keepdim=True |
| 02.05 | [tiny_gpt/normalization.py](../tiny_gpt/normalization.py) · `LayerNorm.forward` | 减均值后除以 sqrt(var+eps)，eps 在根号内 |
| 02.06 | [tiny_gpt/normalization.py](../tiny_gpt/normalization.py) · `LayerNorm.forward` | 逐元素乘 weight 加 bias；不能用 @ |
| 03.01 | [tiny_gpt/masks.py](../tiny_gpt/masks.py) · `causal_mask` | torch.ones 创建 bool [T,T]，再保留 diagonal=1 的上三角 |
| 03.02 | [tiny_gpt/masks.py](../tiny_gpt/masks.py) · `causal_mask` | 连续补两个前导轴：[T,T]→[1,1,T,T] |
| 04.01 | [tiny_gpt/attention.py](../tiny_gpt/attention.py) · `CausalSelfAttention.split_heads` | 先 reshape(B,T,H,Dh)，再交换 T、H；尺寸是整数 |
| 04.02 | [tiny_gpt/attention.py](../tiny_gpt/attention.py) · `CausalSelfAttention.merge_heads` | 先交换 H、T，再 reshape(B,T,H*Dh) |
| 04.03 | [tiny_gpt/attention.py](../tiny_gpt/attention.py) · `CausalSelfAttention.forward` | 通过 q_proj 投影输入 |
| 04.04 | [tiny_gpt/attention.py](../tiny_gpt/attention.py) · `CausalSelfAttention.forward` | 通过 k_proj 投影输入 |
| 04.05 | [tiny_gpt/attention.py](../tiny_gpt/attention.py) · `CausalSelfAttention.forward` | 通过 v_proj 投影输入 |
| 04.06 | [tiny_gpt/attention.py](../tiny_gpt/attention.py) · `CausalSelfAttention.forward` | QKᵀ/sqrt(Dh)，仅交换 K 的最后两个轴 |
| 04.07 | [tiny_gpt/attention.py](../tiny_gpt/attention.py) · `CausalSelfAttention.forward` | 沿 key 轴 dim=-1 做 softmax |
| 04.08 | [tiny_gpt/attention.py](../tiny_gpt/attention.py) · `CausalSelfAttention.forward` | weights @ v，加权汇总 value |
| 04.09 | [tiny_gpt/attention.py](../tiny_gpt/attention.py) · `CausalSelfAttention.forward` | 合头之后必须经过 out_proj |
| 04.10 | [tiny_gpt/attention.py](../tiny_gpt/attention.py) · `CausalSelfAttention.forward` | 调用 split_heads(q)，得到 [B,H,T,Dh] |
| 04.11 | [tiny_gpt/attention.py](../tiny_gpt/attention.py) · `CausalSelfAttention.forward` | 调用 split_heads(k) |
| 04.12 | [tiny_gpt/attention.py](../tiny_gpt/attention.py) · `CausalSelfAttention.forward` | 调用 split_heads(v) |
| 04.13 | [tiny_gpt/attention.py](../tiny_gpt/attention.py) · `CausalSelfAttention.forward` | masked_fill(blocked,-inf)，True 位置概率应为 0 |
| 05.01 | [tiny_gpt/mlp.py](../tiny_gpt/mlp.py) · `MLP.forward` | 先调用 up 扩维 |
| 05.02 | [tiny_gpt/mlp.py](../tiny_gpt/mlp.py) · `MLP.forward` | GELU 非线性，approximate="tanh" |
| 05.03 | [tiny_gpt/mlp.py](../tiny_gpt/mlp.py) · `MLP.forward` | 调用 down 回到 D |
| 05.04 | [tiny_gpt/block.py](../tiny_gpt/block.py) · `TransformerBlock.forward` | 对输入 x 使用 norm1 |
| 05.05 | [tiny_gpt/block.py](../tiny_gpt/block.py) · `TransformerBlock.forward` | 归一化结果输入 attention |
| 05.06 | [tiny_gpt/block.py](../tiny_gpt/block.py) · `TransformerBlock.forward` | 原始 x 加 attention 输出 |
| 05.07 | [tiny_gpt/block.py](../tiny_gpt/block.py) · `TransformerBlock.forward` | 第一次残差结果输入 norm2 |
| 05.08 | [tiny_gpt/block.py](../tiny_gpt/block.py) · `TransformerBlock.forward` | 输入 mlp |
| 05.09 | [tiny_gpt/block.py](../tiny_gpt/block.py) · `TransformerBlock.forward` | 第一次残差结果加 mlp 输出 |
| 06.01 | [tiny_gpt/model.py](../tiny_gpt/model.py) · `GPT.__init__` | 创建 GPTEmbedding |
| 06.02 | [tiny_gpt/model.py](../tiny_gpt/model.py) · `GPT.__init__` | ModuleList 中独立创建每一层 Block |
| 06.03 | [tiny_gpt/model.py](../tiny_gpt/model.py) · `GPT.__init__` | 创建最终 LayerNorm |
| 06.04 | [tiny_gpt/model.py](../tiny_gpt/model.py) · `GPT.__init__` | Linear(D,V,bias=False)，V 是词表大小 |
| 06.05 | [tiny_gpt/model.py](../tiny_gpt/model.py) · `GPT.forward` | 输入 embedding |
| 06.06 | [tiny_gpt/model.py](../tiny_gpt/model.py) · `GPT.forward` | 循环内调用当前 block 更新 x |
| 06.07 | [tiny_gpt/model.py](../tiny_gpt/model.py) · `GPT.forward` | 所有 block 后再归一化 |
| 06.08 | [tiny_gpt/model.py](../tiny_gpt/model.py) · `GPT.forward` | lm_head 输出 [B,T,V]，不做 softmax |
| 07.01 | [training/data.py](../training/data.py) · `TokenDataset.__len__` | 步长 1，长度为 N-T |
| 07.02 | [training/data.py](../training/data.py) · `TokenDataset.__getitem__` | 从 index 取 T 个 token |
| 07.03 | [training/data.py](../training/data.py) · `TokenDataset.__getitem__` | 起点后移一位，仍取 T 个 token |
| 08.01 | [training/trainer.py](../training/trainer.py) · `train_step` | 先清空上次梯度 |
| 08.02 | [training/trainer.py](../training/trainer.py) · `train_step` | model(x) 得到 [B,T,V] |
| 08.03 | [training/trainer.py](../training/trainer.py) · `train_step` | reshape(-1,V) |
| 08.04 | [training/trainer.py](../training/trainer.py) · `train_step` | reshape(-1) |
| 08.05 | [training/trainer.py](../training/trainer.py) · `train_step` | 交叉熵接收原始分数和 long 标签 |
| 08.06 | [training/trainer.py](../training/trainer.py) · `train_step` | 从标量 loss 反向传播 |
| 08.07 | [training/trainer.py](../training/trainer.py) · `train_step` | 根据梯度更新参数 |
| 09.01 | [training/trainer.py](../training/trainer.py) · `evaluate` | 使用 eval 和 no_grad；按目标 token 数加权求 loss；finally 恢复原训练模式；接口与例子见教程 09 |
| 10.01 | [training/trainer.py](../training/trainer.py) · `clip_gradients` | clip_grad_norm_，使用 error_if_nonfinite=True |
| 10.02 | [training/trainer.py](../training/trainer.py) · `train_accumulated_step` | 整组开始只清空一次 |
| 10.03 | [training/trainer.py](../training/trainer.py) · `train_accumulated_step` | 对当前小 batch 前向 |
| 10.04 | [training/trainer.py](../training/trainer.py) · `train_accumulated_step` | 展开 logits 和 y，计算平均交叉熵 |
| 10.05 | [training/trainer.py](../training/trainer.py) · `train_accumulated_step` | 按当前目标 token 数/整组 token 数缩放 loss |
| 10.06 | [training/trainer.py](../training/trainer.py) · `train_accumulated_step` | 每个小 batch 独立 backward，梯度累积 |
| 10.07 | [training/trainer.py](../training/trainer.py) · `train_accumulated_step` | 整组最后只更新一次 |
| 11.01 | [training/checkpoint.py](../training/checkpoint.py) · `save_checkpoint` | model.state_dict() |
| 11.02 | [training/checkpoint.py](../training/checkpoint.py) · `save_checkpoint` | optimizer.state_dict() |
| 11.03 | [training/checkpoint.py](../training/checkpoint.py) · `save_checkpoint` | 写入已提供的同目录临时路径；原子替换框架已提供 |
| 11.04 | [training/checkpoint.py](../training/checkpoint.py) · `load_checkpoint` | torch.load，map_location="cpu"，weights_only=True |
| 11.05 | [training/checkpoint.py](../training/checkpoint.py) · `load_checkpoint` | 读取 epoch |
| 11.06 | [training/checkpoint.py](../training/checkpoint.py) · `load_checkpoint` | 读取 global_step |
| 11.07 | [training/checkpoint.py](../training/checkpoint.py) · `load_checkpoint` | 把参数复制到已创建的模型 |
| 11.08 | [training/checkpoint.py](../training/checkpoint.py) · `load_checkpoint` | 恢复 Adam 动量和计数 |
| 12.01 | [training/schedule.py](../training/schedule.py) · `get_lr` | 预热：base_lr*step/warmup_steps |
| 12.02 | [training/schedule.py](../training/schedule.py) · `get_lr` | 计划末尾返回最小学习率 |
| 12.03 | [training/schedule.py](../training/schedule.py) · `get_lr` | 衰减进度，扣除预热阶段 |
| 12.04 | [training/schedule.py](../training/schedule.py) · `get_lr` | 从 base_lr 线性插值到 min_lr |
| 12.05 | [training/schedule.py](../training/schedule.py) · `set_lr` | 修改每个参数组字典，不是 optimizer.lr |
| 13.01 | [data_pipeline/tokenizer.py](../data_pipeline/tokenizer.py) · `CharTokenizer.__init__` | 校验词表、复制 id_to_token、用 enumerate 建立反向字典；接口见教程 13 |
| 13.02 | [data_pipeline/tokenizer.py](../data_pipeline/tokenizer.py) · `CharTokenizer.from_text` | 只用训练文本；sorted(set(text))；前置 <unk>，然后 cls(tokens) |
| 13.03 | [data_pipeline/tokenizer.py](../data_pipeline/tokenizer.py) · `CharTokenizer.encode` | 逐字符查字典 get(character,unk_id)，返回 list[int] |
| 13.04 | [data_pipeline/tokenizer.py](../data_pipeline/tokenizer.py) · `CharTokenizer.decode` | 验证编号为合法 Python int，按顺序查表，用空字符串连接 |
| 13.05 | [data_pipeline/tokenizer.py](../data_pipeline/tokenizer.py) · `CharTokenizer.save` | 保存 type/version/tokens 到 UTF-8 JSON，保留编号顺序 |
| 13.06 | [data_pipeline/tokenizer.py](../data_pipeline/tokenizer.py) · `CharTokenizer.load` | 读取 JSON，校验格式，再用原顺序 tokens 重建实例 |
| 14.01 | [training/device.py](../training/device.py) · `resolve_device` | auto 依次检查 CUDA/MPS/CPU；显式设备需校验可用性与编号 |
| 14.02 | [training/device.py](../training/device.py) · `model_device` | 读取第一个模型参数的 device；本项目单设备 |
| 14.03 | [training/device.py](../training/device.py) · `move_batch` | 仅搬当前 batch，保持 long 类型 |
| 15.01 | [data_pipeline/tinystories.py](../data_pipeline/tinystories.py) · `iter_stories` | 按独占一行的 EOS 切分；保留内部换行；跳过空故事；处理尾故事；limit 正数或 None |
| 15.02 | [data_pipeline/tinystories.py](../data_pipeline/tinystories.py) · `train_bpe` | 使用 ByteLevel+BPE，完整字节初始表和 EOS；只从训练集流式训练；返回 tokenizer,count |
| 16.01 | [scripts/prepare_tinystories.py](../scripts/prepare_tinystories.py) · `encode_split` | 批量编码故事，每篇追加 EOS，写 uint16 临时文件；校验编号/大小/EOS 后替换并返回元数据；详见教程 16 |
| 16.02 | [training/binary_data.py](../training/binary_data.py) · `BinaryTokenDataset.__len__` | 固定步长 T，需要 T+1 个 token，尾部不足一组舍弃 |
| 16.03 | [training/binary_data.py](../training/binary_data.py) · `BinaryTokenDataset.__getitem__` | 起点 index*T |
| 16.04 | [training/binary_data.py](../training/binary_data.py) · `BinaryTokenDataset.__getitem__` | 取 T+1 个 token，转换 int64 再转 torch Tensor |
| 16.05 | [training/binary_data.py](../training/binary_data.py) · `BinaryTokenDataset.__getitem__` | 返回错开一位的 x/y |
| 17.01 | [training/run_checkpoints.py](../training/run_checkpoints.py) · `save_run_checkpoint` | 保存历史文件；改善时原子更新 best；原子更新 latest；只删除超出 keep 的历史文件 |
| 18.01 | [scripts/generate_tinystories.py](../scripts/generate_tinystories.py) · `main` | 读取 checkpoint 到 CPU |
| 18.02 | [scripts/generate_tinystories.py](../scripts/generate_tinystories.py) · `main` | 用保存的 model_config 重建 GPTConfig |
| 18.03 | [scripts/generate_tinystories.py](../scripts/generate_tinystories.py) · `main` | 用 cfg 创建 GPT，随后恢复参数 |
| 18.04 | [scripts/generate_tinystories.py](../scripts/generate_tinystories.py) · `main` | tokenizer.encode(prompt)，不添加额外特殊 token |
| 18.05 | [scripts/generate_tinystories.py](../scripts/generate_tinystories.py) · `main` | 每次仅保留最近 max_len 个 token |
| 18.06 | [scripts/generate_tinystories.py](../scripts/generate_tinystories.py) · `main` | 取最后一个位置的分数，得到 [1,V] |
| 18.07 | [scripts/generate_tinystories.py](../scripts/generate_tinystories.py) · `main` | 采样时除以 temperature |
| 18.08 | [scripts/generate_tinystories.py](../scripts/generate_tinystories.py) · `main` | topk 筛选候选，最多 V 个 |
| 18.09 | [scripts/generate_tinystories.py](../scripts/generate_tinystories.py) · `main` | softmax 后 multinomial 抽取一个候选索引 |
| 18.10 | [scripts/generate_tinystories.py](../scripts/generate_tinystories.py) · `main` | 把候选索引映射回实际词表编号 |
| 18.11 | [scripts/generate_tinystories.py](../scripts/generate_tinystories.py) · `main` | top_k=0 时对完整词表采样 |
| 18.12 | [scripts/generate_tinystories.py](../scripts/generate_tinystories.py) · `main` | 贪心时取 argmax 并保留 [1,1] |
| 18.13 | [scripts/generate_tinystories.py](../scripts/generate_tinystories.py) · `main` | 沿序列轴拼接新 token |
| 18.14 | [scripts/generate_tinystories.py](../scripts/generate_tinystories.py) · `main` | 恢复训练好的参数，不能直接使用随机模型 |
