"""因果 mask：bool 中 True 表示禁止关注，False 表示允许关注。

第一版使用等长序列，只实现 causal mask，不处理 padding。
后续由 Attention 内部调用 causal_mask(T, device=x.device)。
"""

from workbook_support import todo

import torch


def causal_mask(length: int, device=None) -> torch.Tensor:
    """返回 bool [1,1,T,T]，其中 T = length。

    第 i 行是正在查询的位置，第 j 列是被关注的位置。
    当 j > i 时禁止关注；允许关注自己和之前的位置。

    length=4 时，最后两个轴应为：
        False  True   True   True
        False  False  True   True
        False  False  False  True
        False  False  False  False

    前两个长度为 1 的轴对应 batch 和 head，后续可广播到
    Attention 分数的 [B,H,T,T]，所有序列、所有头共享同一规则。
    """
    if length <= 0:
        raise ValueError("length 必须大于 0")


    # 实现提示：1：创建形状 [T,T]、元素全为 True 的 bool 张量。
    # 提示：torch.ones；尺寸使用 length；dtype=torch.bool。
    # 将 device 参数传给创建函数，使 mask 与 Attention 分数在同一设备。
    mask = todo('03.01：torch.ones 创建 bool [T,T]，再保留 diagonal=1 的上三角')

    # 实现提示：2：只保留严格上三角为 True，主对角线和下三角为 False。
    # 提示：torch.triu；diagonal=1 表示从主对角线上方一条对角线开始。
    # 不要屏蔽主对角线，因为当前位置允许读取自己。
    mask = torch.triu(mask,diagonal=1)

    # 实现提示：3：在最前面补两个轴，[T,T] -> [1,1,T,T]。
    # 提示：连续两次 unsqueeze(0)，不会改变原有矩阵元素。
    blocked = todo('03.02：连续补两个前导轴：[T,T]→[1,1,T,T]')

    return blocked
