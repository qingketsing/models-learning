"""第 08 关：全参数训练一步；支持语言模型和 Encoder–Decoder。"""
from .objective import token_loss


def train_step(model, optimizer, inputs, targets, pad_id=0, source=None):
    """返回更新前的 loss.item()（Python float）。
    TODO 08a：train 模式 -> zero_grad -> forward -> loss -> backward -> step。
    source=None 时 model(inputs)；否则 model(source, inputs)。
    optimizer 由调用方传入，不能在每一步重新创建。
    不要把 forward 放进 no_grad，也不要在 backward 前 item/detach。
    """
    raise NotImplementedError("TODO 08a：完成训练一步")
