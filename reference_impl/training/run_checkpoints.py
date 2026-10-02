"""长训练保存策略：latest、best、最近 N 个按步数命名的历史文件。"""

import os
from pathlib import Path
import shutil
import tempfile

from .checkpoint import save_checkpoint


def atomic_copy(source, destination):
    destination = Path(destination)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp', dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_run_checkpoint(directory, model, optimizer, epoch, step, config,
                        val_loss, best_val_loss, best_step, improved, keep=3):
    if keep < 1:
        raise ValueError('keep 必须为正')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'step-{step:08d}.pt'
    save_checkpoint(path, model, optimizer, epoch, step, config, extra_state={
        'val_loss': val_loss, 'best_val_loss': best_val_loss, 'best_step': best_step,
    })
    # best 优先更新：latest 发布后，best 文件也已完成保存。
    if improved:
        atomic_copy(path, directory / 'best.pt')
    atomic_copy(path, directory / 'latest.pt')
    for old in sorted(directory.glob('step-????????.pt'))[:-keep]:
        old.unlink()
