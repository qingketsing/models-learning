"""设备迁移、验证、梯度累积及跨设备 checkpoint 恢复的回归检查。"""

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader

from tiny_gpt.config import GPTConfig
from tiny_gpt.model import GPT
from training.data import TokenDataset
from training.device import resolve_device, describe_device
from training.trainer import train_step, train_accumulated_step, evaluate
from training.checkpoint import save_checkpoint, load_checkpoint


def check_optimizer(optimizer):
    for parameter, state in optimizer.state.items():
        for key in ("exp_avg", "exp_avg_sq"):
            assert state[key].device == parameter.device
        # 本项目使用默认非 capturable、非 fused 的 Adam。
        assert state["step"].device.type == "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(42)
    cfg = GPTConfig()
    config = {"test": "cross_device", "model": asdict(cfg)}
    x = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 1]])
    y = torch.tensor([[2, 3, 4, 1], [3, 4, 1, 2]])
    loader = DataLoader(TokenDataset([1, 2, 3, 4] * 4, 4), batch_size=3)

    with tempfile.TemporaryDirectory() as directory:
        cpu_path = Path(directory) / "cpu.pt"
        device_path = Path(directory) / "device.pt"
        cpu_model = GPT(cfg)
        cpu_optimizer = torch.optim.Adam(cpu_model.parameters(), lr=0.001)
        train_step(cpu_model, cpu_optimizer, x, y, max_grad_norm=1.0)
        save_checkpoint(cpu_path, cpu_model, cpu_optimizer, 1, 1, config)

        model = GPT(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        assert load_checkpoint(cpu_path, model, optimizer, config) == (1, 1)
        check_optimizer(optimizer)
        with torch.no_grad():
            torch.testing.assert_close(
                model(x.to(device)).cpu(), cpu_model(x), atol=2e-4, rtol=2e-4
            )
        print("通过：CPU checkpoint 恢复至目标设备，参数与 Adam 状态正确")

        # 输入一直保留为 CPU；函数内部负责搬运，单步和累积都要覆盖。
        before = model.lm_head.weight.detach().clone()
        assert math.isfinite(train_step(model, optimizer, x, y, max_grad_norm=1.0))
        assert math.isfinite(train_accumulated_step(
            model, optimizer, [(x[:1], y[:1]), (x, y)], max_grad_norm=1.0
        ))
        assert not torch.equal(before, model.lm_head.weight)
        assert x.device.type == y.device.type == loader.dataset.tokens.device.type == "cpu"
        check_optimizer(optimizer)
        print("通过：CPU 数据驱动目标设备单步训练及不等大小 batch 梯度累积")

        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        for training in (True, False):
            model.train(training)
            assert math.isfinite(evaluate(model, loader))
            assert model.training == training
        for name, p in model.named_parameters():
            torch.testing.assert_close(before[name], p, rtol=0, atol=0)
        print("通过：验证自动搬运数据，参数不变，调用前模式恢复")

        save_checkpoint(device_path, model, optimizer, 2, 3, config)
        # 先在同一目标设备恢复后继续更新，验证节点上的实际 resume 路径。
        resumed = GPT(cfg).to(device)
        resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=0.001)
        assert load_checkpoint(device_path, resumed, resumed_optimizer, config) == (2, 3)
        check_optimizer(resumed_optimizer)
        assert math.isfinite(train_step(resumed, resumed_optimizer, x, y))

        restored_cpu = GPT(cfg)
        restored_optimizer = torch.optim.Adam(restored_cpu.parameters(), lr=0.001)
        assert load_checkpoint(device_path, restored_cpu, restored_optimizer, config) == (2, 3)
        check_optimizer(restored_optimizer)
        with torch.no_grad():
            torch.testing.assert_close(
                restored_cpu(x), model(x.to(device)).cpu(), atol=2e-4, rtol=2e-4
            )
        assert math.isfinite(train_step(restored_cpu, restored_optimizer, x, y))
        print("通过：目标设备存档可在同设备和 CPU 恢复并继续更新")
    print(f"设备检查全部通过：{describe_device(device)}；未保留临时 checkpoint")


if __name__ == "__main__":
    main()
