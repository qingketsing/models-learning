"""单设备训练工具：模型先迁移，再创建优化器；数据按 batch 搬运。"""

from workbook_support import todo

import torch


def resolve_device(requested="auto"):
    """auto 优先 CUDA，其次 Mac MPS，否则 CPU；显式设备不可用时直接报错。"""
    todo('14.01：auto 依次检查 CUDA/MPS/CPU；显式设备需校验可用性与编号')


def model_device(model):
    """本项目是单设备模型，以参数所在设备为准。"""
    return todo('14.02：读取第一个模型参数的 device；本项目单设备')


def move_batch(x, y, device):
    """只搬运当前 batch，保留 long 类型，不修改 Dataset 中的 CPU 张量。"""
    return todo('14.03：仅搬当前 batch，保持 long 类型')


def describe_device(device):
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device)})"
    return str(device)
