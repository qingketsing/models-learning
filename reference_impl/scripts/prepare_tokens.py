"""第三阶段第 3 关：文本 -> token 张量 -> 可重复加载的数据文件。

运行：uv run python -m scripts.prepare_tokens
只需完成 prepare_token_data 中的 3 个 TODO；文件读写、检查与记录已提供。
本关保留文本中的换行，不添加 EOS，不在这里构造 x/y 或 batch。
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from data_pipeline.tokenizer import CharTokenizer


def prepare_token_data(train_text, val_text):
    """返回 (tokenizer, train_tokens, val_tokens)，张量均为一维 CPU long。"""
    if not train_text or not val_text:
        raise ValueError("训练文本和验证文本都不能为空")

    # 完成 TODO 1～3 后删除这一行。
    # raise NotImplementedError("TODO：构造 tokenizer 并编码文本")

    # TODO 1：只使用 train_text 创建 tokenizer。
    # 提示：调用你写好的 CharTokenizer.from_text(...)。
    tokenizer = CharTokenizer.from_text(train_text)

    # TODO 2：用同一个 tokenizer 编码两份文本，得到两个整数列表。
    # 提示：tokenizer.encode(...)；不要单独为验证集创建词表。
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    # TODO 3：把两个列表分别转换为 torch.long 张量，形状 [N_train]、[N_val]。
    # 提示：torch.tensor(整数列表, dtype=torch.long)。
    # 编号不是可训练参数，不需要 requires_grad，也不需要补 batch 轴。
    train_tokens = torch.tensor(train_ids)
    val_tokens = torch.tensor(val_ids)
    return tokenizer, train_tokens, val_tokens


def validate_tokens(tokens, text, tokenizer):
    """检查存储格式、编号范围和编码内容。"""
    assert isinstance(tokens, torch.Tensor)
    assert tokens.dtype == torch.long and tokens.ndim == 1
    assert tokens.device.type == "cpu"
    assert tokens.numel() == len(text) > 0
    assert bool(((tokens >= 0) & (tokens < tokenizer.vocab_size)).all())
    assert tokens.tolist() == tokenizer.encode(text)


def main():
    parser = argparse.ArgumentParser(description="准备字符级 token 缓存")
    parser.add_argument("--text-dir", default="data/text")
    parser.add_argument("--output-dir", default="data/tokenized")
    args = parser.parse_args()
    text_dir = Path(args.text_dir)
    output_dir = Path(args.output_dir)
    if text_dir.resolve() == output_dir.resolve():
        parser.error("输出目录应与文本目录不同，避免覆盖文本的 metadata.json")

    train_path, val_path = text_dir / "train.txt", text_dir / "val.txt"
    train_text = train_path.read_text(encoding="utf-8")
    val_text = val_path.read_text(encoding="utf-8")
    tokenizer, train_tokens, val_tokens = prepare_token_data(train_text, val_text)
    validate_tokens(train_tokens, train_text, tokenizer)
    validate_tokens(val_tokens, val_text, tokenizer)
    assert tokenizer.id_to_token == [tokenizer.unk_token] + sorted(set(train_text))
    assert tokenizer.decode(train_tokens.tolist()) == train_text
    assert not bool((train_tokens == tokenizer.unk_id).any())

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    torch.save(train_tokens, output_dir / "train_tokens.pt")
    torch.save(val_tokens, output_dir / "val_tokens.pt")

    # 重读磁盘文件，检查保存加载没有改变编号含义或张量内容。
    restored = CharTokenizer.load(tokenizer_path)
    assert restored.id_to_token == tokenizer.id_to_token
    for name, expected in [("train_tokens.pt", train_tokens), ("val_tokens.pt", val_tokens)]:
        loaded = torch.load(output_dir / name, map_location="cpu", weights_only=True)
        assert torch.equal(loaded, expected) and loaded.dtype == torch.long

    unknown_count = int((val_tokens == tokenizer.unk_id).sum().item())
    metadata = {
        "format_version": 1,
        "tokenizer_type": "char",
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "vocab_size": tokenizer.vocab_size,
        "dtype": "int64",
        "train_token_count": train_tokens.numel(),
        "val_token_count": val_tokens.numel(),
        "val_unknown_count": unknown_count,
        "val_unknown_ratio": unknown_count / val_tokens.numel(),
        "sources": {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in [("train", train_path), ("val", val_path)]
        },
        "sequence_policy": "continuous text, preserve paragraph newlines, no added EOS",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"词表大小：{tokenizer.vocab_size}（含 <unk>）")
    print(f"训练 tokens：{train_tokens.numel()}，验证 tokens：{val_tokens.numel()}")
    print(f"验证未知字符：{unknown_count}，比例：{metadata['val_unknown_ratio']:.2%}")
    print(f"已保存到：{output_dir}，编码和文件重载检查通过")


if __name__ == "__main__":
    main()
