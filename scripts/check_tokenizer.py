"""检查字符级 tokenizer：uv run python -m scripts.check_tokenizer。"""

import tempfile
from pathlib import Path

from data_pipeline.tokenizer import CharTokenizer


def main():
    text = "我爱学习。\n我 爱模型！"
    tokenizer = CharTokenizer.from_text(text)
    assert tokenizer.id_to_token == ["<unk>"] + sorted(set(text))
    assert tokenizer.vocab_size == len(set(text)) + 1
    assert tokenizer.token_to_id["<unk>"] == 0
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert len(tokenizer.encode(text)) == len(text)
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""
    assert tokenizer.encode("🦊") == [0]
    assert tokenizer.decode([0]) == "<unk>"
    reordered = CharTokenizer.from_text(text[::-1])
    assert reordered.token_to_id == tokenizer.token_to_id
    for invalid in [[-1], [tokenizer.vocab_size], [1.5]]:
        try:
            tokenizer.decode(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"未拒绝非法编号：{invalid}")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "nested" / "tokenizer.json"
        tokenizer.save(path)
        restored = CharTokenizer.load(path)
        assert restored.id_to_token == tokenizer.id_to_token
        assert restored.encode(text) == tokenizer.encode(text)
        assert restored.decode(restored.encode(text)) == text
        # 刻意使用未排序词表，确认 load 保留保存时的编号。
        custom = CharTokenizer(["<unk>", "我", "。", "爱"])
        custom.save(path)
        assert CharTokenizer.load(path).id_to_token == custom.id_to_token
    print("通过：词表稳定性、往返转换、空白保留、未知字符、非法编号、保存加载")

    train_path = Path("data/text/train.txt")
    val_path = Path("data/text/val.txt")
    if train_path.exists() and val_path.exists():
        train_text = train_path.read_text(encoding="utf-8")
        val_text = val_path.read_text(encoding="utf-8")
        actual = CharTokenizer.from_text(train_text)
        assert actual.decode(actual.encode(train_text)) == train_text
        val_ids = actual.encode(val_text)
        unknown = val_ids.count(actual.unk_id)
        ratio = unknown / len(val_ids) if val_ids else 0.0
        print(f"真实训练文本往返通过，词表大小：{actual.vocab_size}")
        print(f"验证字符数：{len(val_ids)}，未知字符数：{unknown}，比例：{ratio:.2%}")
        print("未知字符比例是数据观察指标，不要求为零；不要用验证文本扩建词表。")


if __name__ == "__main__":
    main()
