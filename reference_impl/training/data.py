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
        # TODO 1：返回可以构造的样本数量
        # 提示：token 总数减去输入序列长度
        # raise NotImplementedError("TODO 1：计算样本数量")
        return len(self.tokens) - self.seq_len

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("样本索引越界")

        # 完成下面 TODO 后删除这一行
        # raise NotImplementedError("TODO 2～3：构造输入和目标")

        # TODO 2：从 index 开始取 seq_len 个 token
        # 输出形状：[T]
        x = self.tokens[index : index + self.seq_len]

        # TODO 3：起点比 x 后移一位，同样取 seq_len 个 token
        # 输出形状：[T]
        y = self.tokens[index+1 : index + self.seq_len + 1]

        return x, y
