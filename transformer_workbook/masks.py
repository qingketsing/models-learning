"""第 03 关。全工作册统一：bool mask 中 True 表示【禁止关注】。"""
import torch


def causal_mask(length, device=None):
    """返回 [1,1,T,T] 的 bool 严格上三角；允许看自己。
    TODO 03a：先创建 [T,T]，再用 triu(diagonal=1)，补两个轴。
    第 i 行是 query，第 j 列是 key，j>i 的位置为 True。
    """
    raise NotImplementedError("TODO 03a：未来位置 mask")


def padding_mask(ids, pad_id=0):
    """long [B,S] -> bool [B,1,1,S]，只屏蔽 key 位置。
    TODO 03b：比较 ids == pad_id，再补 head/query 两个轴。
    padding query 的损失另行忽略；不要在这里屏蔽整行 query。
    """
    raise NotImplementedError("TODO 03b：padding key mask")


def decoder_mask(ids, pad_id=0):
    """返回 [B,1,T,T]。输入只支持右 padding，且每条至少一个有效 token。
    TODO 03c：将前两个函数的结果用逻辑或 | 合并。
    """
    if ids.ndim != 2 or ids.size(1) == 0:
        raise ValueError("需要非空 [B,T]")
    valid = ids.ne(pad_id)
    if not valid[:, 0].all() or ((~valid[:, :-1]) & valid[:, 1:]).any():
        raise ValueError("只支持右 padding；每条开头必须有有效 token/BOS")
    raise NotImplementedError("TODO 03c：合并 causal 与 padding mask")
