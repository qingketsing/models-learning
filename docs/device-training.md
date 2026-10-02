# 单设备训练

训练入口支持 `--device auto|cpu|mps|cuda|cuda:N`。
`auto` 优先 CUDA，其次 Apple MPS，否则 CPU。显式请求不可用的设备会报错。
模型先移动到目标设备，再创建优化器；Dataset 留在 CPU，训练与验证每次
只搬运当前小 batch。梯度累积不会预先把整组数据移到显存。

## 本机

```bash
uv run python -m scripts.check_device --device cpu
uv run python -m scripts.check_device --device mps
uv run python -m scripts.train_text --device cpu --check-only
```

## 当前算力节点

连接：

```bash
ssh -p <端口> <用户名>@<算力节点地址>
cd /root/autodl-tmp/models-learning-device-20261002
```

已探测到一张 NVIDIA GeForce RTX 4090（约 24 GiB），节点现有环境为
Python 3.12.3、PyTorch 2.12.1+cu130，解释器是 `/root/miniconda3/bin/python`。
使用该解释器，不需要重新安装。本地 pyproject.toml 的 torch 来源绑定 CPU
索引，因此没有把本地依赖锁文件用于节点环境，也没有在节点运行 uv sync。

```bash
/root/miniconda3/bin/python -m scripts.check_device --device cuda
/root/miniconda3/bin/python -m scripts.train_text --device cuda --check-only
/root/miniconda3/bin/python -m scripts.train_text --device cuda --epochs 5
```

默认保存到节点工作目录下的 `checkpoints/text.pt`。恢复时保持数据和训练配置一致：

```bash
/root/miniconda3/bin/python -m scripts.train_text --device cuda --epochs 5 --resume checkpoints/text.pt
```

`--epochs` 是目标总轮数，不是追加轮数。CPU 与 GPU 存档可跨设备加载，
设备不计入必须相同的训练配置。恢复前必须先将模型移到设备，再创建优化器。
加载时先把文件读取到 CPU，PyTorch 再将权重和 Adam 动量恢复到参数所在设备。
默认 Adam 的标量 step 留在 CPU 是正常行为，不需要强行迁移所有状态张量。

## 验证范围

`scripts.check_device` 验证单步训练、梯度累积、验证模式恢复、参数更新、
CPU 到目标设备的恢复、目标设备到 CPU 的恢复，以及加载后继续训练。
临时测试 checkpoint 在临时目录中使用，不覆盖训练存档。

这是单设备全精度实现，尚未加入 AMP、分布式训练或精确随机状态恢复。
节点上的小规模试跑使用现有教学文本，不能当作正式数据集训练的效果评估。
