from workbook_support import todo

# tiny_gpt/embeddings.py

import torch
from torch import nn

from .config import GPTConfig


class GPTEmbedding(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.max_len = cfg.max_len

        # 实现提示：1：token 表，参数形状 [V,D]
        self.token = todo('01.01：创建 nn.Embedding(V,D)，它是模块，不是直接参与 @ 的矩阵')
        # 实现提示：2：可学习的位置表，参数形状 [max_len,D]
        self.position = todo('01.02：创建 nn.Embedding(max_len,D)，位置参数也参与训练')


    def forward(self, input_ids):
        """输入 long [B,T]，输出 float [B,T,D]。"""
        if input_ids.ndim != 2:
            raise ValueError("input_ids 必须是 [B,T]")

        if input_ids.dtype != torch.long:
            raise ValueError("input_ids 必须是 torch.long")

        B, T = input_ids.shape

        if not 0 < T <= self.max_len:
            raise ValueError("输入长度超出允许范围")

        # 实现提示：3：查 token 表，得到 [B,T,D]
        token_vectors = todo('01.03：调用 token 查表，得到 [B,T,D]')

        # 实现提示：4：生成位置编号 [T]，device 与输入一致
        positions = todo('01.04：torch.arange(T)，设备与 input_ids 相同')

        # 实现提示：5：查位置表，补 batch 轴，得到 [1,T,D]
        position_vectors = todo('01.05：查位置表并补 batch 轴：[T,D]→[1,T,D]')

        # 实现提示：6：相加，得到 [B,T,D]
        return todo('01.06：相加并广播，返回 [B,T,D]')
