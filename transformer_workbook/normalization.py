"""第 02 关：手写 LayerNorm，禁止用 nn.LayerNorm 替代本关。"""
import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        # 每个特征一个参数，所有 batch/token 共享，不能写成 [B,T,D]。
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        """[B,T,D] -> [B,T,D]。

        TODO 02：只沿 D 求均值、总体方差，再归一化与仿射变换。
        公式：(x-mean)/sqrt(var+eps) * weight + bias。
        mean/var 保留最后一轴：keepdim=True，得到 [B,T,1]。
        torch.var 的 correction 必须是 0；也可以手写平方差的平均。
        eps 放在根号内。提示见 HINTS.md 02。
        """
        raise NotImplementedError("TODO 02：实现 LayerNorm")
