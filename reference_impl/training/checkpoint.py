"""练习：在一轮训练结束后保存，并在新进程中恢复训练。

支持跨设备恢复；保存模型、Adam 状态、完成的 epoch、累计更新次数和配置。
暂不保存随机数和数据迭代器状态，因此不保证与不中断训练逐步一致。
"""

from dataclasses import asdict
from pathlib import Path

import os
import tempfile

import torch


def save_checkpoint(path, model, optimizer, epoch, global_step, train_config, *, extra_state=None):
    """epoch 表示已经完成的轮次；每轮结束后调用。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 完成 TODO 1～3 后删除这一行。
    # raise NotImplementedError("TODO：保存 checkpoint")

    # TODO 1：获取模型参数字典。提示：model.state_dict()。
    # 字典包含各层权重和 buffer；不要保存整个 model 对象。
    model_state = model.state_dict()

    # TODO 2：获取优化器状态字典。提示：optimizer.state_dict()。
    # Adam 还保存了梯度的一阶、二阶矩估计和更新次数，不能只存学习率。
    optimizer_state = optimizer.state_dict()

    # 已提供：用普通字典组织状态，配置也转换成普通字典。
    checkpoint = {
        "model": model_state,
        "optimizer": optimizer_state,
        "epoch": epoch,
        "global_step": global_step,
        "model_config": asdict(model.cfg),
        "train_config": train_config,
    }

    # TODO 3：将 checkpoint 字典写入 path。
    # 提示：torch.save(要保存的对象, 文件路径)。
    if extra_state is not None:
        checkpoint["extra_state"] = extra_state
    # 同目录临时文件 + 原子替换：写入失败时保留旧 checkpoint。
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path, model, optimizer, train_config):
    """模型先 .to(device)，再创建优化器，最后恢复。返回 (epoch, global_step)。

    文件先读取到 CPU。model.load_state_dict 复制到模型的目标设备；
    optimizer.load_state_dict 根据参数设备迁移 Adam 的动量状态。
    普通 Adam 的 step 计数保持在 CPU，这是 PyTorch 的预期行为，
    不要无差别把优化器中所有 Tensor 强制移到 GPU。
    """
    # 完成 TODO 4～8 后删除这一行。
    # raise NotImplementedError("TODO：恢复 checkpoint")

    # TODO 4：读取字典到 CPU。
    # 提示：torch.load(path, map_location="cpu", weights_only=True)。
    checkpoint = torch.load(path,map_location="cpu",weights_only=True)

    # 已提供：防止误用不匹配的模型或数据设置。
    if checkpoint["model_config"] != asdict(model.cfg):
        raise ValueError("模型配置与 checkpoint 不一致")
    if checkpoint["train_config"] != train_config:
        raise ValueError("训练配置与 checkpoint 不一致")

    # TODO 5：恢复模型参数。
    # 提示：model.load_state_dict(checkpoint["model"])。
    model.load_state_dict(checkpoint["model"])

    # TODO 6：恢复优化器状态。
    # 提示：optimizer.load_state_dict(...)，读取字典的 optimizer 项。
    optimizer.load_state_dict(checkpoint["optimizer"])

    # TODO 7：读取已经完成的 epoch 和累计更新次数。
    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]

    # TODO 8：返回两个值，入口会从 epoch + 1 继续。
    return epoch, global_step
