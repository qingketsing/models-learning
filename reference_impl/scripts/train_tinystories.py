"""单卡 TinyStories 训练入口：固定更新预算，复用已有梯度累积与验证。"""

import argparse
from itertools import islice
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from data_pipeline.tinystories import sha256
from tiny_gpt.config import GPTConfig
from tiny_gpt.model import GPT
from training.binary_data import BinaryTokenDataset
from training.checkpoint import save_checkpoint, load_checkpoint
from training.run_checkpoints import save_run_checkpoint
from training.device import resolve_device, describe_device
from training.schedule import get_lr, set_lr
from training.trainer import train_accumulated_step, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=300, help="目标累计更新次数")
    p.add_argument("--schedule-steps", type=int, default=300)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--run-dir", help="独立实验目录：latest、best、最近几个历史 checkpoint 和指标")
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accumulation-steps", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-batch-size", type=int, help="验证 batch 独立于训练；默认与训练相同")
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--checkpoint", default="checkpoints/tinystories.pt")
    p.add_argument("--resume")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.accumulation_steps, args.seq_len,
           args.eval_every, args.eval_batches, args.keep_checkpoints) <= 0:
        p.error("计数必须为正")
    if not 0 < args.warmup_steps < args.schedule_steps or not args.steps <= args.schedule_steps:
        p.error("需要 0 < warmup-steps < schedule-steps，且 steps 不超过 schedule-steps")
    if not 0 <= args.min_lr <= args.lr or not math.isfinite(args.lr) or args.lr <= 0:
        p.error("需要 0 <= min-lr <= lr，且 lr 为有限正数")
    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.resume and Path(args.resume).resolve() != (run_dir / "latest.pt").resolve():
            p.error("长训练请从同一实验目录的 latest.pt 恢复")
        if not args.resume and (any(run_dir.glob("*.pt")) or (run_dir / "metrics.jsonl").exists()):
            p.error("实验已有训练结果：请使用新目录，或显式 --resume")
    elif Path(args.checkpoint).exists() and not args.resume and not args.check_only:
        p.error("checkpoint 已存在：请换输出路径或显式 --resume")
    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
    if eval_batch_size <= 0:
        p.error("eval-batch-size 必须为正")
    torch.manual_seed(42)
    device = resolve_device(args.device)
    root = Path(args.data_dir)
    meta = json.loads((root / "metadata.json").read_text())
    if meta["dtype"] != "<u2" or sha256(root / "tokenizer.json") != meta["tokenizer_sha256"]:
        raise ValueError("数据格式或 tokenizer 校验失败")
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    assert tokenizer.get_vocab_size() == meta["vocab_size"]
    datasets = {}
    for split in ("train", "val"):
        path = root / f"{split}.bin"
        if sha256(path) != meta["splits"][split]["sha256"]:
            raise ValueError(f"{split} 文件校验失败")
        datasets[split] = BinaryTokenDataset(path, args.seq_len, meta["splits"][split]["tokens"])
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=eval_batch_size, shuffle=False)
    cfg = GPTConfig(vocab_size=meta["vocab_size"], max_len=args.seq_len,
                    d_model=256, n_heads=4, n_layers=6, d_ff=1024)
    model = GPT(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    config = {
        "batch_size": args.batch_size, "accumulation_steps": args.accumulation_steps,
        "seq_len": args.seq_len, "optimizer": "Adam", "max_grad_norm": 1.0,
        "schedule": dict(base_lr=args.lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps, total_steps=args.schedule_steps),
        "tokenizer_sha256": meta["tokenizer_sha256"],
        "data_sha256": {s: meta["splits"][s]["sha256"] for s in ("train", "val")},
    }
    if run_dir:
        config["validation"] = {"eval_batches": args.eval_batches, "eval_every": args.eval_every}
        if eval_batch_size != args.batch_size:
            config["validation"]["eval_batch_size"] = eval_batch_size
    print(f"设备：{describe_device(device)}，参数量：{sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"训练块：{len(datasets['train'])}，验证块：{len(datasets['val'])}", flush=True)
    x, y = next(iter(val_loader))
    with torch.no_grad():
        out = model(x.to(device))
        assert out.shape == (*x.shape, cfg.vocab_size) and torch.isfinite(out).all()
    print(f"前向检查通过：{tuple(x.shape)} -> {tuple(out.shape)}", flush=True)
    del out
    if args.check_only:
        return
    completed_epochs, step = 0, 0
    best_val_loss, best_step = float("inf"), 0
    if args.resume:
        completed_epochs, step = load_checkpoint(args.resume, model, optimizer, config)
        state = torch.load(args.resume, map_location="cpu", weights_only=True).get("extra_state", {})
        if run_dir and "best_val_loss" not in state:
            raise ValueError("长训练恢复需要包含最佳验证记录的 checkpoint；旧模型请使用原入口配置恢复")
        best_val_loss = state.get("best_val_loss", float("inf"))
        best_step = state.get("best_step", 0)
        if run_dir:
            if not (run_dir / "best.pt").is_file():
                raise ValueError("缺少 best.pt；恢复时请保留整个实验目录")
            with (run_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps({"kind": "resume", "step": step,
                    "note": "此前高于恢复步数的未保存更新作废；数据重新打乱"}) + "\n")
        print(f"已恢复 global_step={step}；数据从新一轮打乱开始，不保证逐步复现。", flush=True)
    if step >= args.steps:
        raise ValueError(f"checkpoint 已到 {step} 步，目标 steps={args.steps} 无剩余更新")
    if run_dir:
        (run_dir / "config.json").write_text(json.dumps({"model": vars(cfg), "training": config,
            "target_steps": args.steps, "data_dir": str(root.resolve())}, indent=2) + "\n")
    print(f"学习率计划：{config['schedule']}，目标更新次数：{args.steps}", flush=True)
    iterator = iter(train_loader)
    while step < args.steps:
        batches = list(islice(iterator, args.accumulation_steps))
        if not batches:
            completed_epochs += 1
            iterator = iter(train_loader)
            continue
        lr = get_lr(step + 1, **config["schedule"])
        set_lr(optimizer, lr)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        loss = train_accumulated_step(model, optimizer, batches, max_grad_norm=1.0)
        if not math.isfinite(loss):
            raise FloatingPointError("训练 loss 非有限值，停止且不覆盖已有 checkpoint")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        step += 1
        tokens = sum(y.numel() for _, y in batches)
        print(f"step={step}, loss={loss:.4f}, lr={lr:.6g}, update_time={elapsed:.3f}s, compute_tokens/s={tokens/elapsed:.0f}", flush=True)
        if run_dir:
            with (run_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps({"kind": "train", "step": step, "loss": loss,
                    "lr": lr, "tokens": tokens, "update_seconds": elapsed}) + "\n")
        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, islice(val_loader, args.eval_batches))
            if not math.isfinite(val_loss):
                raise FloatingPointError("验证 loss 非有限值，停止且不覆盖已有 checkpoint")
            if run_dir:
                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss, best_step = val_loss, step
                save_run_checkpoint(run_dir, model, optimizer, completed_epochs, step, config,
                    val_loss, best_val_loss, best_step, improved, args.keep_checkpoints)
                with (run_dir / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps({"kind": "validation", "step": step, "val_loss": val_loss,
                        "best_val_loss": best_val_loss, "best_step": best_step}) + "\n")
                saved = run_dir / "latest.pt"
            else:
                save_checkpoint(args.checkpoint, model, optimizer, completed_epochs, step, config)
                saved = args.checkpoint
            print(f"val_loss={val_loss:.4f} (固定前 {args.eval_batches} 个 batch，实际取可用数量), saved={saved}", flush=True)
    print(f"训练完成：global_step={step}", flush=True)


if __name__ == "__main__":
    main()
