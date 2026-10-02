"""加载 TinyStories checkpoint，逐 token 续写。默认贪心，便于比较训练效果。"""

import argparse
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from data_pipeline.tinystories import EOS, sha256
from tiny_gpt.config import GPTConfig
from tiny_gpt.model import GPT
from training.device import resolve_device

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(PROJECT / 'runs/tinystories-10k-v1/best.pt'))
    parser.add_argument('--tokenizer', default='/root/autodl-tmp/datasets/TinyStories/bpe8k-full/tokenizer.json')
    parser.add_argument('--prompt', default='Once upon a time')
    parser.add_argument('--max-new-tokens', type=int, default=150)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--sample', action='store_true', help='启用 temperature + top-k 随机采样')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=40, help='0 表示不筛选候选 token')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.top_k < 0:
        parser.error('max-new-tokens 必须为正，top-k 不能为负')
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('temperature 必须为有限正数')
    if not args.prompt.strip() or EOS in args.prompt:
        parser.error('请输入非空普通文本，不包含 EOS 特殊标记')

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if sha256(args.tokenizer) != checkpoint['train_config']['tokenizer_sha256']:
        raise ValueError('tokenizer 与训练时不一致，不能混用 token 编号')
    tokenizer = Tokenizer.from_file(args.tokenizer)
    cfg = GPTConfig(**checkpoint['model_config'])
    if tokenizer.get_vocab_size() != cfg.vocab_size:
        raise ValueError('词表大小与模型不一致')
    eos_id = tokenizer.token_to_id(EOS)
    if eos_id is None:
        raise ValueError('tokenizer 缺少 EOS')

    model = GPT(cfg)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    step = checkpoint['global_step']
    del checkpoint

    ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)  # [1,T]
    if len(ids) > cfg.max_len:
        print(f'提示词超过 {cfg.max_len} tokens：每次预测只读取最近 {cfg.max_len} 个。')
    mode = f'采样 temperature={args.temperature}, top_k={args.top_k}' if args.sample else '贪心选择'
    print(f'checkpoint：{args.checkpoint}\n训练步数：{step}，设备：{device}\n生成方式：{mode}', flush=True)
    reason, generated = '达到最大生成长度', 0
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            # 学习位置表最多容纳 max_len 个位置，超出时使用滑动窗口。
            context = input_ids[:, -cfg.max_len:]
            logits = model(context)[:, -1, :]  # [1,V]：最后位置对下一个 token 的预测
            if not torch.isfinite(logits).all():
                raise FloatingPointError('模型输出包含 NaN/Inf')
            if args.sample:
                logits = logits / args.temperature
                if args.top_k:
                    values, indices = torch.topk(logits, min(args.top_k, cfg.vocab_size), dim=-1)
                    selected = torch.multinomial(torch.softmax(values, dim=-1), num_samples=1)
                    next_id = indices.gather(-1, selected)
                else:
                    next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)  # [1,1]
            if next_id.item() == eos_id:
                reason = '模型生成了 EOS（故事结束）'
                break
            input_ids = torch.cat([input_ids, next_id], dim=1)
            generated += 1
    print('\n生成结果：\n' + tokenizer.decode(input_ids[0].tolist()))
    print(f'\n新增 {generated} 个非 EOS token；停止原因：{reason}')


if __name__ == '__main__':
    main()
