"""第 08 关续：每次只追加一个 token，重新前向，不使用 KV cache。"""
import torch


@torch.no_grad()
def greedy_generate(model, prompt, max_new_tokens, eos_id=2):
    """prompt 是无 PAD 的 long [1,T]；返回包含 prompt 的 [1,T+新增长度]。
    TODO 08b：保存 training 状态 -> eval -> 循环 -> 恢复原状态。
    每轮 model(ids)，只取最后位置 logits[:, -1, :]，argmax 后保留 [1,1]，
    torch.cat 沿 dim=1 追加。追加 EOS 后立即停止。
    达到 model.cfg.max_len 也停止；max_new_tokens=0 时不改变输入。
    提示：try/finally 内 model.train(was_training) 恢复模式。
    只支持单条生成，不偷偷裁剪上下文，也不实现 padding batch 生成。
    """
    if prompt.ndim != 2 or prompt.size(0) != 1 or prompt.size(1) == 0:
        raise ValueError("本练习只支持单条非空 prompt [1,T]")
    if prompt.dtype != torch.long or prompt.eq(model.cfg.pad_id).any():
        raise ValueError("prompt 必须 long 且不含 PAD")
    if max_new_tokens < 0 or prompt.size(1) > model.cfg.max_len:
        raise ValueError("生成长度不合法")
    if prompt[0, -1].item() == eos_id:
        return prompt.clone()
    raise NotImplementedError("TODO 08b：逐个生成 token")
