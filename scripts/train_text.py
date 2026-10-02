"""第三阶段第 4 关：把真实文本 token 接入现有模型和训练系统。

先填 4 个 TODO，再运行 uv run python -m scripts.train_text --check-only。
检查通过后，去掉 --check-only 即可做小规模文本训练。
本关复用滑动窗口 Dataset；暂不改变 stride、padding 或文档边界策略。
"""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_pipeline.tokenizer import CharTokenizer
from tiny_gpt.config import GPTConfig
from tiny_gpt.model import GPT
from training.data import TokenDataset
from training.trainer import train_epoch_accumulated, evaluate
from training.schedule import get_lr
from training.checkpoint import save_checkpoint, load_checkpoint
from training.device import resolve_device, describe_device, move_batch


def build_components(tokenizer, train_tokens, val_tokens, seq_len, batch_size):
    """接好 Dataset、DataLoader 与 GPT。"""
    # 完成 TODO 1～4 后删除这一行。
    # raise NotImplementedError("TODO：将真实 token 接入 Dataset 和模型")

    # TODO 1：分别创建训练和验证 Dataset，使用传入的 seq_len。
    # 提示：TokenDataset(一维 token 张量, seq_len=seq_len)。
    # 不要提前错位标签；Dataset.__getitem__ 已经负责构造 x、y。
    train_dataset = TokenDataset(train_tokens,seq_len=seq_len)
    val_dataset = TokenDataset(val_tokens,seq_len=seq_len)

    # TODO 2：创建两个 DataLoader，使用传入的 batch_size。
    # 提示：DataLoader(dataset, batch_size=batch_size, shuffle=...)。
    # 训练使用 shuffle=True，验证使用 shuffle=False。
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

    # TODO 3：创建 GPTConfig，词表大小必须来自 tokenizer。
    # 提示：GPTConfig(vocab_size=tokenizer.vocab_size, max_len=seq_len)。
    # 其他参数先保留小模型默认值；不能沿用玩具词表大小 10。
    cfg = GPTConfig(vocab_size=tokenizer.vocab_size,max_len=seq_len)

    # TODO 4：根据 cfg 创建新的 GPT 模型，不能加载旧的数字玩具模型参数。
    model = GPT(cfg)
    return model, train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="字符级真实文本训练")
    parser.add_argument("--device", default="auto", help="auto / cpu / mps / cuda / cuda:N")
    parser.add_argument("--data-dir", default="data/tokenized")
    parser.add_argument("--check-only", action="store_true", help="只检查数据与前向计算，不更新参数")
    parser.add_argument("--epochs", type=int, default=5, help="目标总轮数")
    parser.add_argument("--schedule-steps", type=int, default=115, help="固定学习率计划，恢复时保持不变")
    parser.add_argument("--checkpoint", default="checkpoints/text.pt")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    if args.epochs <= 0 or args.schedule_steps <= 10:
        parser.error("epochs 必须为正，schedule-steps 必须大于 10")

    torch.manual_seed(42)
    device = resolve_device(args.device)
    root = Path(args.data_dir)
    tokenizer_path = root / "tokenizer.json"
    tokenizer = CharTokenizer.load(tokenizer_path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    if tokenizer_hash != metadata["tokenizer_sha256"] or tokenizer.vocab_size != metadata["vocab_size"]:
        raise ValueError("tokenizer 与数据记录不匹配，请重新准备 token 数据")

    train_tokens = torch.load(root / "train_tokens.pt", map_location="cpu", weights_only=True)
    val_tokens = torch.load(root / "val_tokens.pt", map_location="cpu", weights_only=True)
    seq_len, batch_size, accumulation_steps = 16, 8, 4
    for name, tokens in [("train", train_tokens), ("val", val_tokens)]:
        assert tokens.ndim == 1 and tokens.dtype == torch.long
        assert tokens.numel() > seq_len
        assert tokens.numel() == metadata[f"{name}_token_count"]
        assert bool(((tokens >= 0) & (tokens < tokenizer.vocab_size)).all())

    model, train_loader, val_loader = build_components(
        tokenizer, train_tokens, val_tokens, seq_len, batch_size
    )
    # 必须在创建优化器之前迁移模型。
    model = model.to(device)
    # 用验证 batch 检查形状，避免提前消耗训练 loader 的随机打乱状态。
    x, y = next(iter(val_loader))
    x, y = move_batch(x, y, device)
    assert x.shape == y.shape and x.shape[1] == seq_len
    assert torch.equal(x[:, 1:], y[:, :-1])
    with torch.no_grad():
        logits = model(x)
        assert logits.shape == (*x.shape, tokenizer.vocab_size)
        loss = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
        assert torch.isfinite(logits).all() and torch.isfinite(loss)
    updates_per_epoch = (len(train_loader) + accumulation_steps - 1) // accumulation_steps
    print(f"设备：{describe_device(device)}")
    print(f"词表：{tokenizer.vocab_size}，x/y：{tuple(x.shape)}，logits：{tuple(logits.shape)}")
    print(f"训练样本：{len(train_loader.dataset)}，验证样本：{len(val_loader.dataset)}")
    print(f"每轮小 batch：{len(train_loader)}，参数更新：{updates_per_epoch}")
    print(f"验证未知 token 比例：{(val_tokens == tokenizer.unk_id).float().mean().item():.2%}")
    if args.check_only:
        print("数据接入检查通过；没有训练或写入 checkpoint。")
        return

    # 以下为已完成的训练系统，仅更换数据、配置和 checkpoint 路径。
    schedule = dict(base_lr=0.003, min_lr=0.0003, warmup_steps=10, total_steps=args.schedule_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=schedule["base_lr"])
    train_config = {
        "seq_len": seq_len, "batch_size": batch_size,
        "accumulation_steps": accumulation_steps, "max_grad_norm": 1.0,
        "optimizer": "Adam", "schedule": {"kind": "linear_warmup_decay", **schedule},
        "tokenizer_tokens": tokenizer.id_to_token,
        "tokenizer_sha256": tokenizer_hash,
        "data_sha256": {
            split: hashlib.sha256((root / f"{split}_tokens.pt").read_bytes()).hexdigest()
            for split in ("train", "val")
        },
    }
    start_epoch, global_step = 1, 0
    if args.resume:
        completed, global_step = load_checkpoint(args.resume, model, optimizer, train_config)
        start_epoch = completed + 1
        print(f"恢复：已完成 {completed} 轮、{global_step} 次参数更新")
    print(f"训练前 val_loss={evaluate(model, val_loader):.4f}")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, global_step = train_epoch_accumulated(
            model, optimizer, train_loader, accumulation_steps=accumulation_steps,
            global_step=global_step, lr_fn=partial(get_lr, **schedule), max_grad_norm=1.0,
        )
        val_loss = evaluate(model, val_loader)
        print(f"epoch={epoch}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        save_checkpoint(args.checkpoint, model, optimizer, epoch, global_step, train_config)
        print(f"已保存：{args.checkpoint}，global_step={global_step}")


if __name__ == "__main__":
    main()
