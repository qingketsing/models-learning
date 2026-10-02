from workbook_support import todo

from itertools import islice

import torch
import torch.nn.functional as F

from .schedule import set_lr
from .device import model_device, move_batch


def clip_gradients(model, max_grad_norm):
    """原地裁剪参数梯度，返回裁剪前的全局 L2 范数（Python 数字）。

    必须在 backward 后、optimizer.step 前调用；不裁剪参数本身。
    """
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm 必须大于 0")


    # 实现提示：1：调用 torch.nn.utils.clip_grad_norm_ 原地裁剪梯度。
    # 参数：model.parameters()、max_norm=max_grad_norm、error_if_nonfinite=True。
    # 默认 norm_type=2，计算所有参数梯度合在一起的 L2 范数。
    # 返回的是裁剪前的总范数，不是裁剪后的梯度或模型。
    grad_norm = todo('10.01：clip_grad_norm_，使用 error_if_nonfinite=True')

    # 实现提示：2：把标量 Tensor 转成 Python 数字，方便后续记录日志。
    return grad_norm.item()


def train_step(model, optimizer, x, y, max_grad_norm=None):
    """完成一次参数更新，返回本次更新前的 loss 数值。

    x: long [B,T]
    y: long [B,T]
    输入可留在 CPU；本函数将当前 batch 搬到模型所在设备。
    """
    model.train()

    x, y = move_batch(x, y, model_device(model))


    # 实现提示：1：清空上一次计算留下的梯度
    # 提示：调用 optimizer.zero_grad()
    todo('08.01：先清空上次梯度')

    # 实现提示：2：将 x 输入模型，得到 [B,T,V] 的原始分数
    logits = todo('08.02：model(x) 得到 [B,T,V]')

    # 已提供：从输出中读取词表大小
    vocab_size = logits.size(-1)

    # 实现提示：3：展开 batch 和位置两个轴
    # [B,T,V] -> [B*T,V]
    # 提示：reshape(-1, vocab_size)
    flat_logits = todo('08.03：reshape(-1,V)')

    # 实现提示：4：按相同顺序展开目标
    # [B,T] -> [B*T]
    # 提示：reshape(-1)
    flat_targets = todo('08.04：reshape(-1)')

    # 实现提示：5：计算交叉熵，得到标量 Tensor
    # 提示：F.cross_entropy(预测分数, 正确编号)
    loss = todo('08.05：交叉熵接收原始分数和 long 标签')

    # 实现提示：6：从 loss 开始反向传播，计算参数梯度
    todo('08.06：从标量 loss 反向传播')

    # 已接好：先计算梯度，再裁剪，最后更新参数。None 表示不裁剪。
    if max_grad_norm is not None:
        clip_gradients(model, max_grad_norm)

    # 实现提示：7：让优化器根据梯度更新参数
    todo('08.07：根据梯度更新参数')

    # 返回 Python 数字，供打印和记录
    return loss.item()

def train_epoch(model, optimizer, loader, global_step=0, lr_fn=None, max_grad_norm=None):
    """遍历一轮训练数据，返回按目标 token 数加权的平均 loss。"""

    total_loss = 0.0
    total_tokens = 0


    for step, (x, y) in enumerate(loader, start=1):
        # global_step 是此前已完成的更新数，step 是本轮从 1 开始的编号。
        # 必须在 train_step 内的 optimizer.step 之前设置本次学习率。
        update_step = global_step + step
        if lr_fn is not None:
            set_lr(optimizer, lr_fn(update_step))
        # x、y 都是 [B,T]
        # Dataset 数据保留在 CPU，train_step 搬运当前 batch。

        # 实现提示：1：调用已经写好的 train_step
        # 返回值是 Python 浮点数
        loss_value = train_step(model, optimizer, x, y, max_grad_norm=max_grad_norm)

        # 实现提示：2：计算这个 batch 有多少个目标 token
        # 提示：y.numel()，等于 B*T
        token_count = y.numel()

        # 实现提示：3：累加这个 batch 的损失总量
        # train_step 返回的是平均 loss，因此需要乘 token_count
        total_loss += loss_value * token_count

        # 实现提示：4：累加目标 token 数
        total_tokens += token_count

        if step % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  step={step}, global_step={update_step}, loss={loss_value:.4f}, lr={lr:.6f}")

    if total_tokens == 0:
        raise ValueError("训练数据为空")

    # 实现提示：5：计算这一轮按 token 数加权的平均损失
    return total_loss / total_tokens


def train_accumulated_step(model, optimizer, batches, max_grad_norm=None):
    """多个小 batch 累积梯度，只更新一次参数，返回按 token 平均的 loss。

    batches 是一组 CPU (x,y)，没有 padding；逐个搬到模型设备。
    分别执行前向和 backward，不保留所有小 batch 的计算图。
    """
    model.train()
    group_tokens = sum(y.numel() for _, y in batches)
    if group_tokens == 0:
        raise ValueError("累积组不能为空")
    total_loss = 0.0


    # 实现提示：A1：整组开始前清空一次梯度，不要放进下面的小 batch 循环。
    todo('10.02：整组开始只清空一次')

    for x, y in batches:
        x, y = move_batch(x, y, model_device(model))
        # 实现提示：A2：前向计算，并按 [B*T,V] 和 [B*T] 计算平均交叉熵。
        # 提示：与 train_step 的前向和 loss 计算一致。
        logits = todo('10.03：对当前小 batch 前向')
        vocab_size = logits.size(-1)
        loss = todo('10.04：展开 logits 和 y，计算平均交叉熵')

        # 实现提示：A3：按该 batch 占整组的 token 比例缩放 loss。
        # 提示：loss * y.numel() / group_tokens。
        # 等大的 4 个 batch 等价于 loss / 4；尾组只有 2 个则是 loss / 2。
        # 不要固定除以 4，否则尾组和大小不同的 batch 会被错误加权。
        scaled_loss = todo('10.05：按当前目标 token 数/整组 token 数缩放 loss')

        # 实现提示：A4：对 scaled_loss 做 backward，梯度会自动累加。
        # 此处不要 zero_grad、step 或裁剪；也不需要 retain_graph=True。
        todo('10.06：每个小 batch 独立 backward，梯度累积')

        # 日志记录未缩放 loss 的总量，不把缩放后的 loss 当平均训练损失。
        total_loss += loss.item() * y.numel()

    # 实现提示：A5：整组梯度累积完后，调用已有的 clip_gradients。
    if max_grad_norm is not None:
        clip_gradients(model,max_grad_norm)

    # 实现提示：A6：整组只调用一次 optimizer.step()。
    todo('10.07：整组最后只更新一次')
    return total_loss / group_tokens


def train_epoch_accumulated(
    model, optimizer, loader, accumulation_steps=4,
    global_step=0, lr_fn=None, max_grad_norm=None,
):
    """已提供分组框架，返回 (平均 loss, 更新后的 global_step)。

    global_step 只统计 optimizer.step 次数，不统计小 batch 次数。
    """
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps 必须大于 0")
    total_loss = 0.0
    total_tokens = 0
    iterator = iter(loader)

    while True:
        # 一次取最多 accumulation_steps 个 CPU batch；最后不足一组也保留。
        # 这里只暂存输入数据，计算图在逐个 backward 后释放。
        batches = list(islice(iterator, accumulation_steps))
        if not batches:
            break

        update_step = global_step + 1
        if lr_fn is not None:
            set_lr(optimizer, lr_fn(update_step))
        loss_value = train_accumulated_step(
            model, optimizer, batches, max_grad_norm=max_grad_norm
        )
        global_step = update_step
        group_tokens = sum(y.numel() for _, y in batches)
        total_loss += loss_value * group_tokens
        total_tokens += group_tokens
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  global_step={global_step}, micro_batches={len(batches)}, "
            f"loss={loss_value:.4f}, lr={lr:.6f}"
        )

    if total_tokens == 0:
        raise ValueError("训练数据为空")
    return total_loss / total_tokens, global_step


def evaluate(model, loader):
    """返回验证集按目标 token 数加权的平均 loss，不更新参数。

    当前无 padding，所有目标都参与损失；逐个 batch 搬到模型设备。
    不要在此调用 backward、optimizer.step 或 zero_grad。
    """
    todo('09.01：使用 eval 和 no_grad；按目标 token 数加权求 loss；finally 恢复原训练模式；接口与例子见教程 09')
