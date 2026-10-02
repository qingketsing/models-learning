# tiny_gpt/embeddings.py

import torch
from torch import nn

from .config import GPTConfig


class GPTEmbedding(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.max_len = cfg.max_len

        # TODO 1：token 表，参数形状 [V,D]
        self.token = nn.Embedding(
            cfg.vocab_size,
            cfg.d_model,
        )
        # TODO 2：可学习的位置表，参数形状 [max_len,D]
        self.position = nn.Embedding(
            cfg.max_len,
            cfg.d_model
        )


    def forward(self, input_ids):
        """输入 long [B,T]，输出 float [B,T,D]。"""
        if input_ids.ndim != 2:
            raise ValueError("input_ids 必须是 [B,T]")

        if input_ids.dtype != torch.long:
            raise ValueError("input_ids 必须是 torch.long")

        B, T = input_ids.shape

        if not 0 < T <= self.max_len:
            raise ValueError("输入长度超出允许范围")

        # TODO 3：查 token 表，得到 [B,T,D]
        token_vectors = self.token(input_ids)

        # TODO 4：生成位置编号 [T]，device 与输入一致
        positions = torch.arange(T, device=input_ids.device)

        # TODO 5：查位置表，补 batch 轴，得到 [1,T,D]
        position_vectors = self.position(positions).unsqueeze(0)

        # TODO 6：相加，得到 [B,T,D]
        return token_vectors + position_vectors