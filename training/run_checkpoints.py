"""长训练保存策略：latest、best、最近 N 个按步数命名的历史文件。"""

from workbook_support import todo

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
    todo('17.01：保存历史文件；改善时原子更新 best；原子更新 latest；只删除超出 keep 的历史文件')
