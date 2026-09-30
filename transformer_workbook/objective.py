"""第 07 关：沿用你刚学过的 token 错位与交叉熵。"""
import torch
import torch.nn.functional as F


def shift_tokens(tokens):
    """long [B,L] -> inputs/targets 均为 [B,L-1]。
    TODO 07a：输入去掉最后一列，标签去掉第一列，不要切 batch 轴。
    数据约定：BOS 内容 EOS PAD...，只在本函数错位一次。
    """
    if tokens.ndim != 2 or tokens.size(1) < 2:
        raise ValueError("至少需要两个 token")
    raise NotImplementedError("TODO 07a：构造错位输入标签")


def token_loss(logits, targets, pad_id=0):
    """[B,T,V] 和 [B,T] -> 标量 []，只对非 PAD 标签取平均。
    TODO 07b：reshape 保留 V；cross_entropy(ignore_index=pad_id)。
    targets 必须 long。用 reshape 兼容切片产生的非连续 Tensor。
    全 PAD 的 batch 没有训练信号，应明确抛错而不是返回 NaN。
    EOS 是有效目标，必须参加 loss，PAD 才忽略。
    """
    if targets.dtype != torch.long or logits.shape[:-1] != targets.shape:
        raise ValueError("要求 logits [B,T,V] 与 long targets [B,T] 对齐")
    if not targets.ne(pad_id).any():
        raise ValueError("没有有效目标：标签不能全是 PAD")
    raise NotImplementedError("TODO 07b：忽略 padding 的交叉熵")
