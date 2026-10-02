"""手写 LayerNorm：只沿每个 token 的特征轴归一化。

本练习不要用 nn.LayerNorm 或 F.layer_norm 代替计算。
接口：LayerNorm(d_model)，输入和输出均为 [B,T,D]。
"""

import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # TODO 1：创建可训练的缩放参数 weight，形状 [D]，初始值全为 1。
        # 提示：用 torch.ones 创建张量，用 nn.Parameter 注册为模型参数。
        # 每个特征一个参数；所有 batch 和 token 共享，不要创建 [B,T,D]。
        self.weight = nn.Parameter(torch.ones(self.d_model))

        # TODO 2：创建可训练的平移参数 bias，形状 [D]，初始值全为 0。
        # 提示：torch.zeros + nn.Parameter。
        self.bias = nn.Parameter(torch.zeros(self.d_model))


    def forward(self, x):
        """[B,T,D] -> [B,T,D]。

        每个 token 独立计算：
        mean = 特征的平均值
        var = (特征 - mean) 的平方的平均值
        output = (x - mean) / sqrt(var + eps) * weight + bias
        """
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError("x 必须是 [B,T,D]，且 D 等于 d_model")

        # 完成下面所有 TODO 后，删除这一行。
        # raise NotImplementedError("TODO：完成 LayerNorm 的参数和前向计算")

        # TODO 3：只沿最后一维 D 求均值，得到 [B,T,1]。
        # 提示：x.mean(...)，使用 dim=-1、keepdim=True。
        mean = x.mean(dim = -1,keepdim = True)

        # TODO 4：求总体方差，得到 [B,T,1]。
        # 提示：先计算 x - mean，再平方，最后沿 D 求平均并保留轴。
        # 也可用 x.var(...)，但必须设置 correction=0，不能用默认值。
        var = x.var(dim=-1, correction=0, keepdim=True)

        # TODO 5：减均值，再除以标准差，得到 [B,T,D]。
        # 提示：torch.sqrt；eps 放在根号里面，防止方差为 0 时除以 0。
        normalized = (x - mean)/torch.sqrt(self.eps + var)

        # TODO 6：乘缩放参数，再加平移参数，得到 [B,T,D]。
        # 提示：[D] 参数会自动广播到每个 batch、每个 token。
        output = normalized * self.weight + self.bias

        return output
