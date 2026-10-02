"""TinyStories 原始故事迭代器和 Byte-level BPE 工具。"""

from workbook_support import todo

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
    todo('15.01：按独占一行的 EOS 切分；保留内部换行；跳过空故事；处理尾故事；limit 正数或 None')


def train_bpe(path, limit=10_000, vocab_size=8000):
    todo('15.02：使用 ByteLevel+BPE，完整字节初始表和 EOS；只从训练集流式训练；返回 tokenizer,count')
