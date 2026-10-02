"""MLP：对每个 token 独立做非线性变换，参数在所有位置共享。"""

from workbook_support import todo

from torch import nn
import torch.nn.functional as F

from .config import GPTConfig


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # 已提供：先扩展特征维度，再投影回原维度。
        self.up = nn.Linear(cfg.d_model, cfg.d_ff)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model)

    def forward(self, x):
        """[B,T,D] -> [B,T,d_ff] -> [B,T,D]。"""

        # 实现提示：1：调用 self.up，将最后一维 D 扩展为 d_ff。
        hidden = todo('05.01：先调用 up 扩维')

        # 实现提示：2：对 hidden 做 GELU 激活，形状不变。
        # 提示：F.gelu(..., approximate="tanh")。
        # 非线性激活让两层 Linear 不只是等价于一次线性变换。
        hidden = todo('05.02：GELU 非线性，approximate="tanh"')

        # 实现提示：3：调用 self.down，将 d_ff 投影回 D。
        # 不要做 softmax，也不要在这里添加残差；残差由 Block 负责。
        output = todo('05.03：调用 down 回到 D')
        return output
