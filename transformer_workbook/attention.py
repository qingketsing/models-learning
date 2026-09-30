"""第 04 关：同一个模块用于 self-attention 与 cross-attention。
不能调用 nn.MultiheadAttention / scaled_dot_product_attention 完成填空。
"""
import math
import torch
from torch import nn


def scaled_attention(q, k, v, blocked=None):
    """q [B,H,Tq,Dh]，k/v [B,H,Tk,Dh]。
    返回 (context [B,H,Tq,Dh], weights [B,H,Tq,Tk])。
    blocked: bool，可广播到分数矩阵；True 表示屏蔽。

    TODO 04a：QK^T / sqrt(Dh)；masked_fill(...,-inf)；softmax；乘 V。
    softmax 沿 key 轴，即最后一轴。不要用 sqrt(Tk) 或 sqrt(D)。
    全屏蔽行要抛 ValueError，避免 softmax([-inf,...]) 产生 NaN。
    提示：blocked.all(dim=-1).any()；本工作册 mask 的 key 轴长度为 Tk。
    """
    raise NotImplementedError("TODO 04a：缩放、mask、softmax、加权求和")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("D 必须能被 H 整除")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # nn.Linear 权重内部是 [out,in]；与以前手写 x@w 的方向不同。
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        """TODO 04b：[B,T,D] -> [B,H,T,Dh]。先 reshape，再 transpose。"""
        raise NotImplementedError("TODO 04b：拆头")

    def merge_heads(self, x):
        """TODO 04c：[B,H,T,Dh] -> [B,T,D]。先交换轴，再 reshape。"""
        raise NotImplementedError("TODO 04c：合并头")

    def forward(self, query, key, value, blocked=None):
        """返回 (output [B,Tq,D], weights [B,H,Tq,Tk])。
        TODO 04d：分别投影 Q/K/V，拆头，调用 04a，合并并执行 out_proj。
        cross-attention 中 Tq 与 Tk 可以不同，不要把它们写成同一个 T。
        注意：先合并各头，再输出投影；不能遗漏 W_o。
        """
        raise NotImplementedError("TODO 04d：组装多头注意力")
