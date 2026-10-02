"""从 TinyStories 训练集的固定前缀训练 BPE；不会读取验证集。"""

import argparse
import json
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer

from data_pipeline.tinystories import EOS, sha256, train_bpe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--vocab-size", type=int, default=8000)
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "tokenizer.json"
    if path.exists():
        raise FileExistsError(f"为避免更改已有 token 编号，请使用新输出目录：{path}")
    tokenizer, count = train_bpe(args.train_file, args.limit, args.vocab_size)
    temporary = root / "tokenizer.json.tmp"
    tokenizer.save(str(temporary))
    loaded = Tokenizer.from_file(str(temporary))
    sample = "A little dog found a friend."
    assert loaded.encode(sample).ids == tokenizer.encode(sample).ids
    temporary.replace(path)
    metadata = {
        "type": "byte_level_bpe", "library_version": tokenizers.__version__,
        "source": str(Path(args.train_file).resolve()), "source_sha256": sha256(args.train_file),
        "selection": "first N nonempty stories in original order; train split only",
        "requested_stories": args.limit, "actual_stories": count,
        "requested_vocab_size": args.vocab_size, "vocab_size": tokenizer.get_vocab_size(),
        "eos_id": tokenizer.token_to_id(EOS), "tokenizer_sha256": sha256(path),
    }
    (root / "tokenizer_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
