"""加载 TinyStories checkpoint，逐 token 续写。默认贪心，便于比较训练效果。"""

from workbook_support import todo

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
    checkpoint = todo('18.01：读取 checkpoint 到 CPU')
    if sha256(args.tokenizer) != checkpoint['train_config']['tokenizer_sha256']:
        raise ValueError('tokenizer 与训练时不一致，不能混用 token 编号')
    tokenizer = Tokenizer.from_file(args.tokenizer)
    cfg = todo('18.02：用保存的 model_config 重建 GPTConfig')
    if tokenizer.get_vocab_size() != cfg.vocab_size:
        raise ValueError('词表大小与模型不一致')
    eos_id = tokenizer.token_to_id(EOS)
    if eos_id is None:
        raise ValueError('tokenizer 缺少 EOS')

    model = todo('18.03：用 cfg 创建 GPT，随后恢复参数')
    todo('18.14：恢复训练好的参数，不能直接使用随机模型')
    model = model.to(device)
    model.eval()
    step = checkpoint['global_step']
    del checkpoint

    ids = todo('18.04：tokenizer.encode(prompt)，不添加额外特殊 token')
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)  # [1,T]
    if len(ids) > cfg.max_len:
        print(f'提示词超过 {cfg.max_len} tokens：每次预测只读取最近 {cfg.max_len} 个。')
    mode = f'采样 temperature={args.temperature}, top_k={args.top_k}' if args.sample else '贪心选择'
    print(f'checkpoint：{args.checkpoint}\n训练步数：{step}，设备：{device}\n生成方式：{mode}', flush=True)
    reason, generated = '达到最大生成长度', 0
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            # 学习位置表最多容纳 max_len 个位置，超出时使用滑动窗口。
            context = todo('18.05：每次仅保留最近 max_len 个 token')
            logits = todo('18.06：取最后一个位置的分数，得到 [1,V]')  # [1,V]：最后位置对下一个 token 的预测
            if not torch.isfinite(logits).all():
                raise FloatingPointError('模型输出包含 NaN/Inf')
            if args.sample:
                logits = todo('18.07：采样时除以 temperature')
                if args.top_k:
                    values, indices = todo('18.08：topk 筛选候选，最多 V 个')
                    selected = todo('18.09：softmax 后 multinomial 抽取一个候选索引')
                    next_id = todo('18.10：把候选索引映射回实际词表编号')
                else:
                    next_id = todo('18.11：top_k=0 时对完整词表采样')
            else:
                next_id = todo('18.12：贪心时取 argmax 并保留 [1,1]')  # [1,1]
            if next_id.item() == eos_id:
                reason = '模型生成了 EOS（故事结束）'
                break
            input_ids = todo('18.13：沿序列轴拼接新 token')
            generated += 1
    print('\n生成结果：\n' + tokenizer.decode(input_ids[0].tolist()))
    print(f'\n新增 {generated} 个非 EOS token；停止原因：{reason}')


if __name__ == '__main__':
    main()
