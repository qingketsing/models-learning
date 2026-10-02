"""Pre-Norm Transformer Block：两次归一化、两个子模块、两次残差。"""

from torch import nn

from .config import GPTConfig
from .normalization import LayerNorm
from .attention import CausalSelfAttention
from .mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # 两个 LayerNorm 各有独立参数。
        self.norm1 = LayerNorm(cfg.d_model)
        self.attention = CausalSelfAttention(cfg)
        self.norm2 = LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        """输入和输出均为 [B,T,D]，下面每个中间张量也是此形状。

        计算顺序：r = x + Attention(LN1(x))
                  output = r + MLP(LN2(r))
        Attention 内部处理 causal mask，默认只返回输出张量。
        """
        # 完成全部 TODO 后删除此行。
        # raise NotImplementedError("TODO Block：连接两个子模块和残差")

        # TODO 1：用 norm1 归一化原始输入 x。
        normalized = self.norm1(x)

        # TODO 2：将 normalized 传给 attention，保留默认 return_weights=False。
        attention_output = self.attention(normalized)

        # TODO 3：将 attention_output 与原始 x 相加，得到第一次残差结果。
        # 注意：加回的是 x，不是 normalized。
        residual = x + attention_output

        # TODO 4：用 norm2 归一化第一次残差结果 residual。
        normalized = self.norm2(residual)

        # TODO 5：将新的 normalized 传给 mlp。
        mlp_output = self.mlp(normalized)

        # TODO 6：将 mlp_output 与 residual 相加。
        # 注意：这次加回 residual，不是最开始的 x。
        output = mlp_output + residual
        return output
