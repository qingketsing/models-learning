"""TinyStories 原始故事迭代器和 Byte-level BPE 工具。"""

import hashlib
from itertools import islice
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS = "<|endoftext|>"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_stories(path, limit=None):
    """按独占一行的 EOS 切故事，保留内部换行，只清除故事首尾空白。"""
    if limit is not None and limit <= 0:
        raise ValueError("limit 必须为正或 None")

    def read():
        lines = []
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip() == EOS:
                    story = "".join(lines).strip()
                    if story:
                        yield story
                    lines = []
                else:
                    lines.append(line)
        story = "".join(lines).strip()
        if story:
            yield story

    yield from (read() if limit is None else islice(read(), limit))


def train_bpe(path, limit=10_000, vocab_size=8000):
    if vocab_size < 257:
        raise ValueError("需要容纳 256 个字节 token 和 EOS")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True,
    )
    count = 0

    def texts():
        nonlocal count
        for text in iter_stories(path, limit):
            count += 1
            yield text

    tokenizer.train_from_iterator(texts(), trainer=trainer)
    if count == 0:
        raise ValueError("训练文件没有有效故事")
    for text in ["Once upon a time, a cat smiled.\n", "中文 café 🦊\t  test"]:
        assert tokenizer.decode(tokenizer.encode(text).ids) == text
    return tokenizer, count
