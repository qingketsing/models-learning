from workbook_support import todo

import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len: int):
        super().__init__()

        # 已提供：统一转换成一维 long Tensor
        self.tokens = torch.as_tensor(
            tokens,
            dtype=torch.long,
        )
        self.seq_len = seq_len

        if self.tokens.ndim != 1:
            raise ValueError("tokens 必须是一维 token 序列")

        if seq_len <= 0:
            raise ValueError("seq_len 必须大于 0")

        if len(self.tokens) < seq_len + 1:
            raise ValueError("至少需要 seq_len + 1 个 token")

    def __len__(self):
        # 实现提示：1：返回可以构造的样本数量
        # 提示：token 总数减去输入序列长度
        return todo('07.01：步长 1，长度为 N-T')

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("样本索引越界")


        # 实现提示：2：从 index 开始取 seq_len 个 token
        # 输出形状：[T]
        x = todo('07.02：从 index 取 T 个 token')

        # 实现提示：3：起点比 x 后移一位，同样取 seq_len 个 token
        # 输出形状：[T]
        y = todo('07.03：起点后移一位，仍取 T 个 token')

        return x, y
