"""玩具数据训练入口：在项目根目录运行 uv run python -m scripts.train。"""

import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader

from tiny_gpt.config import GPTConfig
from tiny_gpt.model import GPT
from training.data import TokenDataset
from training.trainer import train_epoch_accumulated, evaluate
from training.checkpoint import save_checkpoint, load_checkpoint
from training.schedule import get_lr
from training.device import resolve_device, describe_device


def main():
    parser = argparse.ArgumentParser(description="Tiny GPT 玩具数据训练")
    parser.add_argument("--device", default="auto", help="auto / cpu / mps / cuda / cuda:N")
    parser.add_argument("--epochs", type=int, default=5, help="目标总轮数，不是新增轮数")
    parser.add_argument("--checkpoint", default="checkpoints/accumulated.pt", help="每轮结束的保存路径")
    parser.add_argument("--schedule-steps", type=int, default=40, help="学习率计划的总更新次数；恢复时保持不变")
    parser.add_argument("--accumulation-steps", type=int, default=4, help="每次更新最多累积的小 batch 数")
    parser.add_argument("--resume", default=None, help="从指定 checkpoint 恢复")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs 必须大于 0")
    if args.accumulation_steps <= 0:
        parser.error("--accumulation-steps 必须大于 0")
    if args.schedule_steps <= 10:
        parser.error("--schedule-steps 必须大于预热步数 10")

    torch.manual_seed(42)
    device = resolve_device(args.device)

    # 数据保留在 CPU，训练时逐个 batch 搬运。
    all_tokens = [1, 2, 3, 4] * 48
    # 先划分原始序列，再分别构造窗口，避免窗口跨越划分边界。
    # 重复数据仍有相同内容，这里只验证流程，不证明真实泛化能力。
    train_tokens = all_tokens[:128]
    val_tokens = all_tokens[128:]
    seq_len = 8
    batch_size = 4
    epochs = args.epochs
    learning_rate = 0.01
    max_grad_norm = 1.0
    schedule_config = {
        "base_lr": learning_rate,
        "min_lr": 0.001,
        "warmup_steps": 10,
        "total_steps": args.schedule_steps,
    }
    # 将固定配置绑定到 get_lr，训练循环只需传入更新编号。
    lr_fn = partial(get_lr, **schedule_config)

    # Dataset 负责切出 x、y；DataLoader 负责组成 batch。
    # shuffle 只改变样本顺序，不改变样本内部 token 的顺序。
    train_dataset = TokenDataset(train_tokens, seq_len=seq_len)
    val_dataset = TokenDataset(val_tokens, seq_len=seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    cfg = GPTConfig()
    model = GPT(cfg).to(device)

    # 优化器在循环外创建一次，让它保留跨 step、epoch 的状态。
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_config = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "accumulation_steps": args.accumulation_steps,
        "optimizer": "Adam",
        "data": "repeat_1234_x48_split128",
        "schedule": {"kind": "linear_warmup_decay", **schedule_config},
    }
    start_epoch = 1
    global_step = 0
    if args.resume:
        completed_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, train_config
        )
        start_epoch = completed_epoch + 1
        print(f"已恢复：完成 {completed_epoch} 轮，累计更新 {global_step} 次")
        if start_epoch > epochs:
            print("已达到目标总轮数；如需继续，请增大 --epochs。")
            return

    updates_per_epoch = (len(train_loader) + args.accumulation_steps - 1) // args.accumulation_steps
    print(f"设备：{describe_device(device)}，训练样本数：{len(train_dataset)}，每轮小 batch 数：{len(train_loader)}")
    print(f"每组最多累积 {args.accumulation_steps} 个小 batch，每轮参数更新 {updates_per_epoch} 次")
    print(f"验证样本数：{len(val_dataset)}，验证 batch 数：{len(val_loader)}")
    print(f"序列长度：{seq_len}，batch size：{batch_size}，学习率：{learning_rate}")
    print(f"学习率计划：预热 10 步，计划总步数 {args.schedule_steps}，最低学习率 0.001")
    print(f"梯度裁剪：全局 L2 范数上限 {max_grad_norm}")

    for epoch in range(start_epoch, epochs + 1):
        print(f"开始第 {epoch}/{epochs} 轮")
        average_loss, global_step = train_epoch_accumulated(
            model, optimizer, train_loader, global_step=global_step, lr_fn=lr_fn,
            max_grad_norm=max_grad_norm, accumulation_steps=args.accumulation_steps,
        )
        val_loss = evaluate(model, val_loader)
        # train_loss 来自训练过程，val_loss 使用本轮更新完成后的模型。
        print(f"epoch={epoch}, train_loss={average_loss:.4f}, val_loss={val_loss:.4f}")

        # 每轮结束后保存；相同路径会更新为最新一轮的状态。
        save_checkpoint(
            args.checkpoint, model, optimizer, epoch, global_step, train_config
        )
        print(f"已保存：{args.checkpoint}，global_step={global_step}")


if __name__ == "__main__":
    main()
