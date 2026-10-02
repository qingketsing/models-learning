"""逐篇编码为小端 uint16 数据；EOS 每篇追加一次，不保留全部数据在内存。"""

import argparse
import json
from itertools import islice
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer

from data_pipeline.tinystories import EOS, iter_stories, sha256


def encode_split(source, destination, tokenizer, limit=None):
    eos = tokenizer.token_to_id(EOS)
    vocab_size = tokenizer.get_vocab_size()
    if eos is None or vocab_size > 65536:
        raise ValueError("需要 EOS 且词表不大于 65536")
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count, total = 0, 0
    with temporary.open("wb") as output:
        stories = iter(iter_stories(source, limit))
        while batch := list(islice(stories, 512)):
            if any(EOS in story for story in batch):
                raise ValueError("正文包含非分隔行 EOS，请检查源数据")
            # Rust tokenizer 批量并行编码，结果顺序与输入一致；不改变 token 编号。
            encodings = tokenizer.encode_batch(batch, add_special_tokens=False)
            batch_ids = []
            for story, encoding in zip(batch, encodings):
                ids = encoding.ids
                if not ids or min(ids) < 0 or max(ids) >= vocab_size:
                    raise ValueError("无效编码结果")
                if count < 10:
                    assert tokenizer.decode(ids) == story
                ids.append(eos)
                batch_ids.extend(ids)
                count += 1
                total += len(ids)
                if count % 10000 == 0:
                    print(f"{destination.name}: {count} stories, {total} tokens", flush=True)
            output.write(np.asarray(batch_ids, dtype="<u2").tobytes())
    if count == 0:
        raise ValueError("没有有效故事")
    if temporary.stat().st_size != total * 2:
        raise ValueError("文件大小与 token 数不匹配")
    data = np.memmap(temporary, dtype="<u2", mode="r")
    eos_count = 0
    for start in range(0, total, 1_000_000):
        chunk = data[start:start + 1_000_000]
        assert int(chunk.max()) < vocab_size
        eos_count += int(np.count_nonzero(chunk == eos))
    assert eos_count == count and int(data[-1]) == eos
    del data
    temporary.replace(destination)
    return {"stories": count, "tokens": total, "bytes": total * 2,
            "sha256": sha256(destination), "source": str(Path(source).resolve()),
            "source_sha256": sha256(source), "requested_limit": limit}


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
