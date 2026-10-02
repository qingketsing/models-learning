"""手写因果多头自注意力：输入、输出均为 [B,T,D]。

练习中不用 nn.MultiheadAttention 或 scaled_dot_product_attention。
第一版无 dropout、padding、KV cache；Q/K/V 都来自同一个输入 x。
"""

from workbook_support import todo

import math

import torch
from torch import nn

from .config import GPTConfig
from .masks import causal_mask


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")

        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        # 已提供：四个可训练的线性层，在所有位置共享。
        # Linear 是模块，用 self.q_proj(x) 调用，不是直接与模块做 @。
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def split_heads(self, x):
        """[B,T,D] -> [B,H,T,Dh]，其中 D = H * Dh。"""
        B, T, D = x.shape

        # 实现提示：1：先 reshape 为 [B,T,H,Dh]，再交换 T 和 H 轴。
        # 提示：reshape(B, T, self.n_heads, self.head_dim)，然后 transpose。
        # 不要直接 reshape 成 [B,H,T,Dh]，那会打乱位置与头的对应关系。
        split = todo('04.01：先 reshape(B,T,H,Dh)，再交换 T、H；尺寸是整数')
        return split

    def merge_heads(self, x):
        """[B,H,T,Dh] -> [B,T,D]。"""
        B, H, T, Dh = x.shape

        # 实现提示：2：先交换 H 和 T 轴，再 reshape 合并 H、Dh。
        # 提示：目标中间形状 [B,T,H,Dh]，最终形状 [B,T,self.d_model]。
        # transpose 后使用 reshape，避免直接 view 遇到不连续内存报错。
        merged = todo('04.02：先交换 H、T，再 reshape(B,T,H*Dh)')
        return merged

    def forward(self, x, return_weights=False):
        """默认返回 output [B,T,D]，方便后面的 Block 直接调用。

        调试时传 return_weights=True，返回 (output, weights)，
        其中 weights 为 [B,H,T,T]，最后两轴分别是 query、key 位置。
        """
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError("x 必须是 [B,T,D]，且 D 等于 d_model")
        B, T, D = x.shape
        if T == 0:
            raise ValueError("序列长度必须大于 0")


        # 实现提示：3：分别用三个线性层投影 x，得到 q、k、v，各为 [B,T,D]。
        q = todo('04.03：通过 q_proj 投影输入')
        k = todo('04.04：通过 k_proj 投影输入')
        v = todo('04.05：通过 v_proj 投影输入')

        # 实现提示：4：分别调用 split_heads，变成 [B,H,T,Dh]。
        q = todo('04.10：调用 split_heads(q)，得到 [B,H,T,Dh]')
        k = todo('04.11：调用 split_heads(k)')
        v = todo('04.12：调用 split_heads(v)')

        # 实现提示：5：计算 QK^T / sqrt(Dh)，得到 scores [B,H,T,T]。
        # 提示：只交换 k 的最后两个轴，用 transpose(-2, -1)。
        # 缩放除数是 math.sqrt(self.head_dim)，不是 sqrt(T) 或 sqrt(D)。
        scores = todo('04.06：QKᵀ/sqrt(Dh)，仅交换 K 的最后两个轴')

        # 已提供：True 表示禁止关注，广播到所有 batch 和 head。
        blocked = causal_mask(T, device=x.device)  # [1,1,T,T]

        # 实现提示：6：将 blocked=True 位置的分数替换为负无穷。
        # 提示：scores.masked_fill(blocked, float("-inf"))。
        # 不能填成 0，因为 softmax 中 exp(0)=1，并不会屏蔽该位置。
        scores = todo('04.13：masked_fill(blocked,-inf)，True 位置概率应为 0')

        # 实现提示：7：沿 key 轴做 softmax，得到 weights [B,H,T,T]。
        # 提示：torch.softmax，dim=-1；每行概率和为 1，未来位置为 0。
        weights = todo('04.07：沿 key 轴 dim=-1 做 softmax')

        # 实现提示：8：权重矩阵乘 v，得到 context [B,H,T,Dh]。
        # 提示：使用 @，这是对不同 key 位置的 value 向量加权求和。
        context = todo('04.08：weights @ v，加权汇总 value')

        # 实现提示：9：调用 merge_heads 得到 [B,T,D]，再调用 out_proj。
        # 输出投影会组合不同头提供的特征，不能在合并头后直接返回。
        merged = self.merge_heads(context)
        output = todo('04.09：合头之后必须经过 out_proj')

        if return_weights:
            return output, weights
        return output
