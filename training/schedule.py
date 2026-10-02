"""学习率练习：线性 warmup + 线性衰减，按参数更新次数计算。

step 从 1 开始，表示即将执行的第几次参数更新。
例如 base_lr=0.01、min_lr=0.001、warmup_steps=10、total_steps=150：
step=1 -> 0.001；10 -> 0.01；80 -> 0.0055；150 及以后 -> 0.001。
"""

from workbook_support import todo


def get_lr(step, base_lr, min_lr, warmup_steps, total_steps):
    """只计算一个学习率数字，不更新模型或优化器。"""
    if step < 1:
        raise ValueError("step 从 1 开始")
    if not 0 < warmup_steps < total_steps:
        raise ValueError("需要 0 < warmup_steps < total_steps")
    if not 0 <= min_lr <= base_lr:
        raise ValueError("需要 0 <= min_lr <= base_lr")


    if step <= warmup_steps:
        # 实现提示：1：线性预热，从 base_lr / warmup_steps 增长到 base_lr。
        # 提示：base_lr 乘以 step / warmup_steps。
        return todo('12.01：预热：base_lr*step/warmup_steps')

    if step >= total_steps:
        # 实现提示：2：到达计划末尾后，保持最低学习率，不继续减到负数。
        return todo('12.02：计划末尾返回最小学习率')

    # 实现提示：3：计算衰减进度，再在最高与最低学习率之间插值。
    # progress = (step - warmup_steps) / (total_steps - warmup_steps)
    # lr = base_lr + (min_lr - base_lr) * progress
    progress = todo('12.03：衰减进度，扣除预热阶段')
    return todo('12.04：从 base_lr 线性插值到 min_lr')


def set_lr(optimizer, lr):
    """把计算出的学习率写入优化器的所有参数组。"""

    for group in optimizer.param_groups:
        # 实现提示：4：设置字典中 lr 对应的值。
        # 提示：group["lr"] = ...；不能只写 optimizer.lr = ...。
        group["lr"] = todo('12.05：修改每个参数组字典，不是 optimizer.lr')
