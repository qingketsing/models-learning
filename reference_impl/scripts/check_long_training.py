"""检查原子保存、历史保留、最佳模型和恢复状态，不依赖真实数据。"""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from tiny_gpt.config import GPTConfig
from tiny_gpt.model import GPT
from training.checkpoint import save_checkpoint, load_checkpoint
from training.run_checkpoints import save_run_checkpoint
from training.schedule import get_lr


def main():
    torch.manual_seed(42)
    model = GPT(GPTConfig())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model(torch.tensor([[1, 2, 3]])).square().mean().backward()
    optimizer.step()
    config = {'test': True}
    with TemporaryDirectory() as directory:
        root = Path(directory)
        best, best_step = float('inf'), 0
        for step, loss in enumerate([4., 3., 5., 6.], 1):
            improved = loss < best
            if improved:
                best, best_step = loss, step
            save_run_checkpoint(root, model, optimizer, 0, step, config,
                                loss, best, best_step, improved, keep=2)
        assert [p.name for p in sorted(root.glob('step-*.pt'))] == ['step-00000003.pt', 'step-00000004.pt']
        saved_best = torch.load(root/'best.pt', weights_only=True)
        latest = torch.load(root/'latest.pt', weights_only=True)
        assert saved_best['global_step'] == 2
        assert latest['global_step'] == 4 and latest['extra_state']['best_val_loss'] == 3.
        restored = GPT(GPTConfig())
        adam = torch.optim.Adam(restored.parameters())
        assert load_checkpoint(root/'latest.pt', restored, adam, config) == (0, 4)
        for a, b in zip(model.parameters(), restored.parameters()):
            assert torch.equal(a, b)
        assert len(adam.state) == len(optimizer.state) > 0
        before = (root/'latest.pt').read_bytes()
        def fail_save(value, path):
            Path(path).write_bytes(b'incomplete')
            raise OSError('simulated interrupted write')
        with patch('training.checkpoint.torch.save', side_effect=fail_save):
            try:
                save_checkpoint(root/'latest.pt', model, optimizer, 0, 5, config)
            except OSError:
                pass
            else:
                raise AssertionError('应该失败')
        assert (root/'latest.pt').read_bytes() == before
        assert not list(root.glob('*.tmp'))
    schedule = dict(base_lr=3e-4, min_lr=3e-5, warmup_steps=200, total_steps=10000)
    assert get_lr(200, **schedule) == 3e-4
    assert get_lr(10000, **schedule) == 3e-5
    assert get_lr(1, **schedule) < get_lr(200, **schedule)
    assert get_lr(201, **schedule) > get_lr(9999, **schedule)
    print('通过：预热/衰减、latest/best、历史保留、参数与优化器恢复、写入失败保留旧文件')


if __name__ == '__main__':
    main()
