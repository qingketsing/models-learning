# 从头开始 TinyStories 长训练

代码与数据都在服务器上操作。默认从随机参数开始，不加载之前 300 步的模型。

## 1. 登录、进入环境

```bash
ssh -p <端口> <用户名>@<算力节点地址>
cd /root/autodl-tmp/models-learning-device-20261002
source .venv/bin/activate
```

## 2. 确认全量编码已经完成

```bash
ls /root/autodl-tmp/datasets/TinyStories/bpe8k-full/metadata.json
```

全量编码已完成并通过校验：

| 划分 | 故事数 | token 数（含 EOS） | 二进制字节数 |
| --- | --- | --- | --- |
| 训练 | 2,119,489 | 459,226,619 | 918,453,238 |
| 验证 | 21,990 | 4,619,199 | 9,238,398 |

词表编号与之前 300 步实验保持一致。只有训练、验证两部分编码与校验全部成功，才会生成这个 metadata 文件。预处理日志：

```bash
tail -n 10 /root/autodl-tmp/datasets/TinyStories/prepare-full.log
```

沿用原有 8,000 词 BPE，不重新训练词表。原始训练和验证文件保持独立，每篇追加一个 EOS。存储使用小端 uint16，Dataset 按需读取并转成 long。短序列跨故事时通过 EOS 表示边界，当前版本不额外阻止跨故事注意力。

优化后的预处理以 512 篇为一批并行编码，只暂存当前批，避免重复查询词表大小；已验证前 10,000 篇的编码 SHA-256 与原版本完全一致。若以后要重建，使用新的输出目录：

```bash
RAYON_NUM_THREADS=8 python -m scripts.prepare_tinystories \
  --raw-dir /root/autodl-tmp/datasets/TinyStories \
  --tokenizer /root/autodl-tmp/datasets/TinyStories/bpe8k/tokenizer.json \
  --output-dir /root/autodl-tmp/datasets/TinyStories/bpe8k-full-new \
  --train-limit 0 --val-limit 0
```

## 3. 从头启动后台训练

```bash
python -m scripts.launch_tinystories --name tinystories-10k-v1
```

这条命令立即返回；监护进程脱离终端运行，SSH 断开不影响训练。无需自己再加 `&`、nohup 或 tmux。机器关机、平台回收或进程被杀仍会停止运行，之后需从 checkpoint 恢复。

实验目录必须是新的，防止覆盖已有结果。如果想另做一次实验，更换 `--name`。之前的 `checkpoints/tinystories.pt` 不受影响。

默认计划：

| 项目 | 值 |
| --- | --- |
| 模型 | D=256，6 层，4 头，FFN=1024，8,900,608 参数 |
| 序列长度 | 256 |
| micro batch / 梯度累积 | 16 / 1（新版启动器默认） |
| 每次完整更新 | 4,096 个目标 token |
| 总更新次数 | 10,000 |
| 学习率 | 前 200 步线性预热至 0.0003，再线性衰减至 0.00003 |
| 精度 / 优化器 | FP32 / Adam |
| 梯度裁剪 | 最大范数 1.0 |
| 验证 / 保存 | 每 500 步及训练最后一步 |
| 验证范围 | 独立 batch=4，固定前 100 个 batch，最多 102,400 个目标 token |
| 历史 checkpoint | 最近 3 份，另保留 latest、best |

10,000 步大约处理 4,096 万目标 token，不等于完整遍历全量数据。`best` 指固定验证子集上损失最低，并非全量验证或人工评估最好的模型。

## 4. 看进度和训练结果

```bash
tail -f runs/tinystories-10k-v1/train.log
```

这里按 Ctrl+C 只停止查看日志，不会停止后台训练。

```bash
cat runs/tinystories-10k-v1/status.json
```

- `running`：监护进程记录的运行状态。
- `completed` 且 `exit_code=0`：训练命令正常结束。
- `failed`：训练命令失败，查看日志末尾。

如机器关机或监护进程被强制终止，状态文件可能停留在 running；它不是实时心跳。可用 `ps -p` 加文件中的 PID 辅助核对进程。重复启动同一实验受文件锁保护。

目录内容：

```text
runs/tinystories-10k-v1/
  train.log             # 终端输出与异常信息
  metrics.jsonl         # 每步 loss、lr、耗时，以及验证记录
  config.json           # 模型、训练计划、数据校验信息
  status.json           # 后台进程状态、启动时间、退出码
  pid                   # 监护进程 PID
  run.lock              # 防止同一实验同时运行
  latest.pt             # 最近一次成功保存
  best.pt               # 固定验证子集 loss 最低的模型
  step-XXXXXXXX.pt      # 最近 3 份历史保存点
```

保存先写临时文件，再原子替换目标文件，降低写入中断损坏旧 checkpoint 的风险。状态文件出现 completed 后，日志最后应为 `训练完成：global_step=10000`。

## 5. 中断后恢复

至少成功保存一次后，运行：

```bash
python -m scripts.launch_tinystories --name tinystories-10k-v1 --resume
```

从该实验 `latest.pt` 恢复模型、优化器、步数与最佳验证记录，继续同一个 10,000 步计划。不要用这条命令加载旧 300 步模型，也不要中途变更学习率计划。恢复前需结束原任务；同一实验只能有一个训练进程。

当前仍不恢复随机数与数据迭代器位置，而是重新打乱数据；不会严格复现不中断的轨迹。最近一次保存之后的更新会丢失。metrics 中会追加 resume 事件，恢复步数之后此前未保存的记录应视为失效。

首次保存前若失败，没有 latest 可恢复，应换一个新实验名从头开始。已经完成 10,000 步后，无需再运行 resume。

## 检查命令

```bash
python -m scripts.check_tinystories
python -m scripts.check_long_training
```

包含：故事边界、BPE 往返、EOS、二进制标签错位、学习率端点、最佳模型、历史保留、模型与优化器恢复，以及模拟写入失败后保留旧 checkpoint。

## 完整遍历一遍数据：新版 batch 配置

节点上的旧训练已停止。启动器新增 `--batch-size`、`--accumulation-steps`、`--eval-batch-size` 参数；默认分别为 16、1、4。训练实际 batch 增大，但有效 batch 仍是 16，验证范围保持原样。独立运行 `scripts.train_tinystories` 的原有训练默认值保留，请显式传参或使用启动器。

```bash
python -m scripts.launch_tinystories \
  --name tinystories-full-epoch1-fast \
  --steps 112116 --warmup-steps 2000 \
  --batch-size 16 --accumulation-steps 1 --eval-batch-size 4 \
  --eval-every 2000 --eval-batches 100
```

这是新目录中的从头训练命令，不会自动恢复旧模型。修改代码本身也不会自动启动这个任务。恢复新版任务需要重复其原有步数与预热等参数并追加 `--resume`。恢复旧 batch=4/累积=4 的任务则必须显式传 `--batch-size 4 --accumulation-steps 4 --eval-batch-size 4`，否则配置校验会拒绝混用。

本次仅调整 batch 参数与启动器接口；模型结构、FP32、优化器和每步计时/同步逻辑未变。
