"""逐篇编码为小端 uint16 数据；EOS 每篇追加一次，不保留全部数据在内存。"""

from workbook_support import todo

import argparse
import json
from itertools import islice
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer

from data_pipeline.tinystories import EOS, iter_stories, sha256


def encode_split(source, destination, tokenizer, limit=None):
    todo('16.01：批量编码故事，每篇追加 EOS，写 uint16 临时文件；校验编号/大小/EOS 后替换并返回元数据；详见教程 16')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=10000, help="0 表示完整训练集")
    parser.add_argument("--val-limit", type=int, default=1000, help="0 表示完整验证集")
    args = parser.parse_args()
    if args.train_limit < 0 or args.val_limit < 0:
        parser.error("limit 不能为负")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name in ["train.bin", "val.bin", "tokenizer.json", "metadata.json"]:
        if (root / name).exists():
            raise FileExistsError(f"请使用新的输出目录，避免覆盖已有数据：{root / name}")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    splits = {}
    for split, name, limit in [("train", "TinyStories-train.txt", args.train_limit),
                                ("val", "TinyStories-valid.txt", args.val_limit)]:
        splits[split] = encode_split(Path(args.raw_dir) / name, root / f"{split}.bin", tokenizer, limit or None)
    shutil.copyfile(args.tokenizer, root / "tokenizer.json")
    metadata = {
        "format_version": 1, "dtype": "<u2", "vocab_size": tokenizer.get_vocab_size(),
        "eos_id": tokenizer.token_to_id(EOS), "tokenizer_sha256": sha256(root / "tokenizer.json"),
        "selection": "first N nonempty stories; official train/validation kept separate",
        "policy": "strip story edges; append one EOS; causal attention may cross story boundaries",
        "splits": splits,
    }
    temporary = root / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(root / "metadata.json")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
