"""第三阶段第 1 关：准备真实文本，先按段落划分，再做后续编码。

在项目根目录运行：uv run python -m scripts.prepare_text
默认输入是本项目新编写的教学语料，仅用于跑通流程，不代表大型语料。
本关没有 tokenizer；不要把段落先切成重叠窗口再随机划分。
"""

import argparse
import hashlib
import json
from pathlib import Path


def read_paragraphs(path):
    """UTF-8 文件 -> 段落列表；约定原始文件用空行分隔段落。"""
    path = Path(path)

    # 读取 UTF-8 文本。
    text = path.read_text(encoding="utf-8")

    # 统一换行符。先把 Windows 的 \r\n 换成 \n，再把 \r 换成 \n。
    # 保留标点、正文中的空格和段落内部换行；不要删除所有空白字符。
    text = text.replace("\r\n","\n")
    text = text.replace("\r","\n")

    # 按空行拆段，保留段落内部空格和换行。
    paragraphs = []
    for part in text.split("\n\n"):
        paragraph = part.strip()
        if paragraph:
            paragraphs.append(paragraph)

    # 已提供：按原顺序去掉完全相同的段落，避免直接重复跨集合出现。
    # 这里只处理完全重复，不能消除近似重复或保证主题独立。
    paragraphs = list(dict.fromkeys(paragraphs))
    if len(paragraphs) < 2:
        raise ValueError("至少需要两个不同的非空段落，请用空行分隔")
    return paragraphs


def split_paragraphs(paragraphs, train_ratio=0.8):
    """按原顺序划分，返回 (训练段落列表, 验证段落列表)。"""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 必须在 0 和 1 之间")
    if len(paragraphs) < 2:
        raise ValueError("至少需要两个段落")

    # 已提供：至少给训练和验证各留一段。
    split_at = max(1, min(len(paragraphs) - 1, int(len(paragraphs) * train_ratio)))

    # 前 split_at 段训练，其余验证。
    train_paragraphs = paragraphs[:split_at]
    val_paragraphs = paragraphs[split_at:]
    return train_paragraphs, val_paragraphs


def write_paragraphs(path, paragraphs):
    """将段落用空行连接并写回 UTF-8 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 用空行连接段落，并在文件末尾保留一个换行。
    text = "\n\n".join(paragraphs) + "\n"
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="按段落准备训练和验证文本")
    parser.add_argument("--input", default="examples/corpus/learning_text.txt")
    parser.add_argument("--output-dir", default="data/text")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    source = Path(args.input)
    output_dir = Path(args.output_dir)
    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"
    if source.resolve() in {train_path.resolve(), val_path.resolve()}:
        parser.error("输出文件不能覆盖原始文本")

    paragraphs = read_paragraphs(source)
    train, val = split_paragraphs(paragraphs, args.train_ratio)
    # 已提供基础边界检查；拼回去应等于原来的完整段落列表。
    assert train and val
    assert train + val == paragraphs
    assert not set(train).intersection(val)

    write_paragraphs(train_path, train)
    write_paragraphs(val_path, val)
    assert train_path.read_text(encoding="utf-8") == "\n\n".join(train) + "\n"
    assert val_path.read_text(encoding="utf-8") == "\n\n".join(val) + "\n"

    # 保存数据来源与处理约定，后续实验才知道用了哪一版材料。
    metadata = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "encoding": "utf-8",
        "cleaning": "normalize newlines, strip paragraph edges, remove empty and exact duplicate paragraphs",
        "split": "ordered paragraphs before tokenization and windowing",
        "requested_train_ratio": args.train_ratio,
        "train_paragraphs": len(train),
        "val_paragraphs": len(val),
        "train_characters": len(train_path.read_text(encoding="utf-8")),
        "val_characters": len(val_path.read_text(encoding="utf-8")),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"有效段落：{len(paragraphs)}，训练段落：{len(train)}，验证段落：{len(val)}")
    print(f"训练文本：{train_path}")
    print(f"验证文本：{val_path}")
    print(f"处理记录：{output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
