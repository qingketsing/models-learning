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

    # 完成下面两个 TODO 后删除此行。
    # raise NotImplementedError("TODO：实现梯度裁剪")

    # TODO 1：调用 torch.nn.utils.clip_grad_norm_ 原地裁剪梯度。
    # 参数：model.parameters()、max_norm=max_grad_norm、error_if_nonfinite=True。
    # 默认 norm_type=2，计算所有参数梯度合在一起的 L2 范数。
    # 返回的是裁剪前的总范数，不是裁剪后的梯度或模型。
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=max_grad_norm,error_if_nonfinite=True)

    # TODO 2：把标量 Tensor 转成 Python 数字，方便后续记录日志。
    return grad_norm.item()


def train_step(model, optimizer, x, y, max_grad_norm=None):
    """完成一次参数更新，返回本次更新前的 loss 数值。

    x: long [B,T]
    y: long [B,T]
    输入可留在 CPU；本函数将当前 batch 搬到模型所在设备。
    """
    model.train()

    x, y = move_batch(x, y, model_device(model))

    # 完成下面 TODO 后删除这一行
    # raise NotImplementedError("TODO：实现一次训练更新")

    # TODO 1：清空上一次计算留下的梯度
    # 提示：调用 optimizer.zero_grad()
    optimizer.zero_grad()

    # TODO 2：将 x 输入模型，得到 [B,T,V] 的原始分数
    logits = model(x)

    # 已提供：从输出中读取词表大小
    vocab_size = logits.size(-1)

    # TODO 3：展开 batch 和位置两个轴
    # [B,T,V] -> [B*T,V]
    # 提示：reshape(-1, vocab_size)
    flat_logits = logits.reshape(-1,vocab_size)

    # TODO 4：按相同顺序展开目标
    # [B,T] -> [B*T]
    # 提示：reshape(-1)
    flat_targets = y.reshape(-1)

    # TODO 5：计算交叉熵，得到标量 Tensor
    # 提示：F.cross_entropy(预测分数, 正确编号)
    loss = F.cross_entropy(flat_logits,flat_targets)

    # TODO 6：从 loss 开始反向传播，计算参数梯度
    loss.backward()

    # 已接好：先计算梯度，再裁剪，最后更新参数。None 表示不裁剪。
    if max_grad_norm is not None:
        clip_gradients(model, max_grad_norm)

    # TODO 7：让优化器根据梯度更新参数
    optimizer.step()

    # 返回 Python 数字，供打印和记录
    return loss.item()

def train_epoch(model, optimizer, loader, global_step=0, lr_fn=None, max_grad_norm=None):
    """遍历一轮训练数据，返回按目标 token 数加权的平均 loss。"""

    total_loss = 0.0
    total_tokens = 0

    # 完成下面 TODO 后删除这一行
    # raise NotImplementedError("TODO：完成一轮训练")

    for step, (x, y) in enumerate(loader, start=1):
        # global_step 是此前已完成的更新数，step 是本轮从 1 开始的编号。
        # 必须在 train_step 内的 optimizer.step 之前设置本次学习率。
        update_step = global_step + step
        if lr_fn is not None:
            set_lr(optimizer, lr_fn(update_step))
        # x、y 都是 [B,T]
        # Dataset 数据保留在 CPU，train_step 搬运当前 batch。

        # TODO 1：调用已经写好的 train_step
        # 返回值是 Python 浮点数
        loss_value = train_step(model, optimizer, x, y, max_grad_norm=max_grad_norm)

        # TODO 2：计算这个 batch 有多少个目标 token
        # 提示：y.numel()，等于 B*T
        token_count = y.numel()

        # TODO 3：累加这个 batch 的损失总量
        # train_step 返回的是平均 loss，因此需要乘 token_count
        total_loss += loss_value * token_count

        # TODO 4：累加目标 token 数
        total_tokens += token_count

        if step % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  step={step}, global_step={update_step}, loss={loss_value:.4f}, lr={lr:.6f}")

    if total_tokens == 0:
        raise ValueError("训练数据为空")

    # TODO 5：计算这一轮按 token 数加权的平均损失
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

    # 完成 TODO A1～A6 后删除这一行。
    # raise NotImplementedError("TODO：实现梯度累积的一次更新")

    # TODO A1：整组开始前清空一次梯度，不要放进下面的小 batch 循环。
    optimizer.zero_grad()

    for x, y in batches:
        x, y = move_batch(x, y, model_device(model))
        # TODO A2：前向计算，并按 [B*T,V] 和 [B*T] 计算平均交叉熵。
        # 提示：与 train_step 的前向和 loss 计算一致。
        logits = model(x)
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(logits.reshape(-1,vocab_size),y.reshape(-1))

        # TODO A3：按该 batch 占整组的 token 比例缩放 loss。
        # 提示：loss * y.numel() / group_tokens。
        # 等大的 4 个 batch 等价于 loss / 4；尾组只有 2 个则是 loss / 2。
        # 不要固定除以 4，否则尾组和大小不同的 batch 会被错误加权。
        scaled_loss = loss * y.numel() / group_tokens

        # TODO A4：对 scaled_loss 做 backward，梯度会自动累加。
        # 此处不要 zero_grad、step 或裁剪；也不需要 retain_graph=True。
        scaled_loss.backward()

        # 日志记录未缩放 loss 的总量，不把缩放后的 loss 当平均训练损失。
        total_loss += loss.item() * y.numel()

    # TODO A5：整组梯度累积完后，调用已有的 clip_gradients。
    if max_grad_norm is not None:
        clip_gradients(model,max_grad_norm)

    # TODO A6：整组只调用一次 optimizer.step()。
    optimizer.step()
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
    was_training = model.training
    total_loss = 0.0
    total_tokens = 0

    try:
        # 完成全部 TODO 后删除这一行。
        # raise NotImplementedError("TODO：完成 evaluate 验证函数")

        # TODO 1：切换到评估模式，提示：model.eval()。
        # eval 切换模块行为，下面的 no_grad 才负责关闭梯度记录。
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                x, y = move_batch(x, y, model_device(model))
                # TODO 2：用 x 做前向计算，得到 logits [B,T,V]。
                logits = model(x)
                vocab_size = logits.size(-1)

                # TODO 3：展开预测和标签，保持位置一一对应。
                # 提示：reshape(-1, vocab_size) 和 reshape(-1)。
                # Dataset 已经做过标签错位，这里不要再次移动标签。
                flat_logits = logits.reshape(-1,vocab_size) # [B*T,V]
                flat_targets = y.reshape(-1)  # [B*T]

                # TODO 4：用 F.cross_entropy 计算标量损失。
                # 传入展开后的原始分数和目标，不要先 softmax。
                loss = F.cross_entropy(flat_logits,flat_targets)

                # TODO 5：按目标 token 数累加损失与计数。
                # 提示：y.numel()；loss.item() * token_count。
                token_count = y.numel()
                total_loss += loss.item() * token_count
                total_tokens += token_count

        if total_tokens == 0:
            raise ValueError("验证数据为空")

        # TODO 6：返回加权平均损失，提示：总损失 / 总 token 数。
        return total_loss / total_tokens
    finally:
        # 即使验证报错，也恢复调用前的模式。
        model.train(was_training)
