"""单设备训练工具：模型先迁移，再创建优化器；数据按 batch 搬运。"""

import torch


def resolve_device(requested="auto"):
    """auto 优先 CUDA，其次 Mac MPS，否则 CPU；显式设备不可用时直接报错。"""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("当前 PyTorch 无法使用 CUDA，请检查节点环境")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA 设备编号 {index} 不存在")
        return torch.device("cuda", index)
    if device.type == "mps":
        if device.index is not None or not torch.backends.mps.is_available():
            raise ValueError("MPS 不可用或设备编号无效，请使用 mps 或 cpu")
        return torch.device("mps")
    if device.type == "cpu":
        return torch.device("cpu")
    raise ValueError("只支持 auto、cpu、mps、cuda 或 cuda:N")


def model_device(model):
    """本项目是单设备模型，以参数所在设备为准。"""
    return next(model.parameters()).device


def move_batch(x, y, device):
    """只搬运当前 batch，保留 long 类型，不修改 Dataset 中的 CPU 张量。"""
    return x.to(device), y.to(device)


def describe_device(device):
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device)})"
    return str(device)
