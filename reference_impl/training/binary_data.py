"""按需读取二进制 token 流，以序列长度为步长构造训练样本。"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BinaryTokenDataset(Dataset):
    def __init__(self, path, seq_len, expected_tokens=None):
        self.path = str(Path(path).resolve())
        self.seq_len = seq_len
        size = Path(path).stat().st_size
        if seq_len <= 0 or size % 2:
            raise ValueError("seq_len 必须为正，uint16 文件必须有偶数字节")
        self.token_count = size // 2
        if expected_tokens is not None and self.token_count != expected_tokens:
            raise ValueError("token 数与元数据不匹配")
        if self.token_count < seq_len + 1:
            raise ValueError("不足一个训练样本")
        self._tokens = None

    def __len__(self):
        return (self.token_count - 1) // self.seq_len

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._tokens is None:
            self._tokens = np.memmap(self.path, dtype="<u2", mode="r")
        start = index * self.seq_len
        chunk = torch.from_numpy(self._tokens[start:start + self.seq_len + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tokens"] = None
        return state
