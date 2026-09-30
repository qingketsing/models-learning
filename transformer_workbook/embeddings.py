"""第 01 关：查表 + 正弦位置编码。提示见 HINTS.md 的 01。"""
import math
import torch
from torch import nn


def sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    """返回 float32 [1,max_len,D]；不需要梯度；D 为正偶数。

    TODO 01a：PE[p,2i]=sin(p/10000**(2i/D))，奇数列改用 cos。
    方法：位置列 [L,1] 与频率行 [D/2] 广播得到 [L,D/2]。
    可以先用双循环写对，再用 arange、切片 0::2 和 1::2 改写。
    检查：位置 0 应为 [0,1,0,1,...]；不要把 batch 编入位置。
    """
    raise NotImplementedError("TODO 01a：创建正弦位置编码表")


class TokenPositionEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, pad_id=0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # buffer 随 model.to(device) 移动，保存进 state_dict，但不是训练参数。
        self.register_buffer("position", sinusoidal_table(max_len, d_model))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: long [B,T] -> float [B,T,D]。

        TODO 01b：self.token(ids) 查表，再加 position 的前 T 个位置。
        本工作册不乘 sqrt(D)，与附带 HTML 的加性位置编码约定一致。
        不要对 token 编号做线性乘法；不要每次 forward 创建 Embedding。
        """
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError("ids 必须是 long [B,T]")
        if not 0 < ids.size(1) <= self.max_len:
            raise ValueError("序列长度必须在 1..max_len 内")
        raise NotImplementedError("TODO 01b：查表并加位置编码")
