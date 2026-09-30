"""第 05 关：GELU FFN 与 Pre-Norm Decoder-only Block。"""
from torch import nn
import torch.nn.functional as F
from .normalization import LayerNorm
from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """TODO 05a：[B,T,D] -> [B,T,F] -> GELU -> [B,T,D]。
        使用 F.gelu(..., approximate='tanh') 与 HTML 激活约定一致。
        参数在 token 间共享，不混合不同位置；不要在此加 softmax。
        """
        raise NotImplementedError("TODO 05a：两层 FFN")


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads)
        self.norm2 = LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff)

    def forward(self, x, blocked):
        """TODO 05b：Pre-Norm，两次残差。
        u=LN1(x)；r=x+Attention(u,u,u)；返回 r+FFN(LN2(r))。
        Attention 返回二元组，残差相加只取 output。
        第二次加回 r，不是最初的 x；本关没有 cross-attention。
        """
        raise NotImplementedError("TODO 05b：Pre-Norm 与残差")
