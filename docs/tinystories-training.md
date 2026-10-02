# TinyStories：BPE、数据预处理和单卡训练

本文记录最初的子集实验。全量数据与新版后台长训练请看 [长训练操作说明](tinystories-long-training.md)。

## 已准备的环境

节点：`ssh -p <端口> <用户名>@<算力节点地址>`

项目目录：`/root/autodl-tmp/models-learning-device-20261002`

进入项目并启用环境：

```bash
cd /root/autodl-tmp/models-learning-device-20261002
source .venv/bin/activate
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

节点虚拟环境继承现有 CUDA PyTorch，仅通过 `requirements-data.txt` 补装数据依赖。不要在节点直接运行 `uv sync`：本地 `pyproject.toml` 配置的是 CPU PyTorch 源。

本地数据依赖安装与检查（直接使用虚拟环境，避免 uv 自动同步移除额外依赖）：

```bash
uv pip install --python .venv/bin/python -r requirements-data.txt
.venv/bin/python -m scripts.check_tinystories
```

## 文件职责

| 文件 | 作用 |
| --- | --- |
| `data_pipeline/tinystories.py` | 按故事分隔符流式读取；训练 byte-level BPE |
| `scripts/train_bpe.py` | 训练、保存 tokenizer 和来源元数据 |
| `scripts/prepare_tinystories.py` | 故事编码、追加 EOS、写二进制与校验元数据 |
| `training/binary_data.py` | 用 memmap 按需读取 token，构造错开一位的 x/y |
| `scripts/train_tinystories.py` | 复用现有模型、梯度累积、裁剪、验证和 checkpoint |
| `scripts/check_tinystories.py` | 小型独立回归检查 |

处理链路：故事文本 → BPE 编号 → 每篇追加一个 EOS → 连续 token 文件 → x/y → GPT → 交叉熵。

BPE 只读取训练集的前 10,000 篇有效故事；验证集只编码、不参与词表训练。256 个字节覆盖基本编码，能往返转换未见过的普通 Unicode 文本。保留故事内部换行，清除故事首尾空白。特殊文本 `<|endoftext|>` 专门作为边界使用。

## 已生成的数据

原始文件位于 `/root/autodl-tmp/datasets/TinyStories`，原始训练、验证文件已完整下载并校验 SHA-256。

- BPE：`/root/autodl-tmp/datasets/TinyStories/bpe8k/tokenizer.json`
- 编码目录：`/root/autodl-tmp/datasets/TinyStories/bpe8k-10k`
- 实际词表：8,000；EOS ID：0。
- 训练：10,000 篇，2,085,669 tokens，4,171,338 字节。
- 验证：1,000 篇，188,920 tokens，377,840 字节。

这是首轮实验子集，尚未编码完整原始数据。选择固定前缀便于复查，不代表随机抽样。验证保持官方独立划分。

二进制存储格式为小端 `uint16`，每个 token 两字节；送入模型时转为 `torch.long`。词表不得超过 65,536。Dataset 以 T 为步长读取 T+1 个编号，x 为前 T 个，y 为后 T 个，不足一个完整样本的尾部舍弃。拼接故事之间使用 EOS，当前版本允许注意力跨故事边界。

## 首次预处理命令（已执行，不必重复）

```bash
python -m scripts.train_bpe \
  --train-file /root/autodl-tmp/datasets/TinyStories/TinyStories-train.txt \
  --output-dir /root/autodl-tmp/datasets/TinyStories/bpe8k \
  --limit 10000 --vocab-size 8000

python -m scripts.prepare_tinystories \
  --raw-dir /root/autodl-tmp/datasets/TinyStories \
  --tokenizer /root/autodl-tmp/datasets/TinyStories/bpe8k/tokenizer.json \
  --output-dir /root/autodl-tmp/datasets/TinyStories/bpe8k-10k \
  --train-limit 10000 --val-limit 1000
```

脚本拒绝覆盖已有输出；重做实验请使用新目录，避免改变 checkpoint 对应的 token 编号。

## 开始第一轮训练

```bash
python -m scripts.train_tinystories \
  --data-dir /root/autodl-tmp/datasets/TinyStories/bpe8k-10k \
  --device cuda --steps 300 --schedule-steps 300 \
  --checkpoint checkpoints/tinystories.pt
```

配置：D=256，6 层，4 头，FFN=1024，序列长度 256，batch=4，累积 4 次；完整更新处理 4,096 个目标 token。第一版使用 FP32，无 AMP。学习率前 10 次更新预热，再线性衰减。

已在 RTX 4090 上通过三次更新的冒烟检查（前两步保存，再恢复执行第三步），模型参数量 8,900,608。输入 `[4,256]`，输出 `[4,256,8000]`；训练 loss 为 9.1609、9.1323、9.0798。检查用 checkpoint 是 `checkpoints/tinystories-smoke.pt`，与正式实验分开。这里只证明流程能够运行，尚未开展正式 300 步训练。

每 50 次更新及最后一次保存 checkpoint，并在验证集固定前 20 个 batch 上计算 loss；这不是完整验证集 loss。数据不足时取全部可用 batch。

恢复训练：在同一命令后追加 `--resume checkpoints/tinystories.pt`。`--steps` 是目标累计步数，不是追加步数。恢复要求模型、数据、tokenizer、batch 和学习率计划一致；恢复模型与优化器，但重新打乱数据，不保证逐步完全复现。延长训练应事先设定更大的 `--schedule-steps`，中途不能直接改变已保存的计划。

## 全量编码（尚未执行）

保持当前 tokenizer 不变，另建数据目录：

```bash
python -m scripts.prepare_tinystories \
  --raw-dir /root/autodl-tmp/datasets/TinyStories \
  --tokenizer /root/autodl-tmp/datasets/TinyStories/bpe8k/tokenizer.json \
  --output-dir /root/autodl-tmp/datasets/TinyStories/bpe8k-full \
  --train-limit 0 --val-limit 0
```

完成后用新 `--data-dir` 和新的 checkpoint 路径启动实验。元数据最后写入，包含源文件、tokenizer、二进制文件的 SHA-256 与 token 数；训练前会核对数据文件。不要把不同 tokenizer 编码的文件混合使用。
