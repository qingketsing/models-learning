"""完成 TODO 后运行的小实验入口；无需填空。"""
import argparse
import torch
from .config import Config
from .data import language_batch, copy_batch, decode
from .model import TinyLanguageModel
from .seq2seq import TinySeq2Seq
from .objective import shift_tokens, token_loss
from .training import train_step
from .generation import greedy_generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["lm", "seq2seq"])
    parser.add_argument("--steps", type=int, default=180)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps 必须大于 0")
    torch.manual_seed(7)
    torch.set_num_threads(1)
    cfg = Config()
    source = None
    if args.mode == "lm":
        model = TinyLanguageModel(cfg)
        inputs, targets = shift_tokens(language_batch())
    else:
        model = TinySeq2Seq(cfg)
        source, tokens = copy_batch()
        inputs, targets = shift_tokens(tokens)
    # 本轮先使用你已理解的 SGD，优化器对照留给原学习计划第五阶段。
    optimizer = torch.optim.SGD(model.parameters(), lr=0.15)
    print("参数数量：", sum(p.numel() for p in model.parameters()))
    for step in range(args.steps):
        loss = train_step(model, optimizer, inputs, targets, cfg.pad_id, source)
        if step % 30 == 0 or step == args.steps - 1:
            print(f"step={step:3d} 更新前 loss={loss:.4f}")
    model.eval()
    with torch.no_grad():
        logits = model(inputs) if source is None else model(source, inputs)
        keep = targets.ne(cfg.pad_id)
        accuracy = logits.argmax(-1).eq(targets)[keep].float().mean().item()
        print("最终训练 loss：", token_loss(logits, targets).item())
        print("训练位置准确率：", accuracy)
    if source is None:
        out = greedy_generate(model, torch.tensor([[1, 3]]), 12, cfg.eos_id)
        print("贪心生成：", decode(out[0]))
    else:
        # 提供推理外壳：训练时的 teacher forcing 与推理时自己追加不同。
        ids = torch.tensor([[cfg.bos_id]])
        with torch.no_grad():
            for _ in range(cfg.max_len - 1):
                next_id = model(source[:1], ids)[:, -1].argmax(-1, keepdim=True)
                ids = torch.cat([ids, next_id], dim=1)
                if next_id.item() == cfg.eos_id:
                    break
        print("源文本：", decode(source[0]))
        print("自主复制：", decode(ids[0]))
    print("这是训练集记忆实验，不是测试集泛化能力证明。")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as error:
        raise SystemExit(f"尚未完成：{error}。请按 README 的关卡顺序填写。")
