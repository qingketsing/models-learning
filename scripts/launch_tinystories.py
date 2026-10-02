"""Linux 节点后台启动器。启动全新 10,000 步实验，或从同一实验 latest 恢复。"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
DATA = Path('/root/autodl-tmp/datasets/TinyStories/bpe8k-full')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='实验名，例如 tinystories-10k-v1')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data-dir', default=str(DATA))
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--warmup-steps', type=int, default=200)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--eval-batches', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--accumulation-steps', type=int, default=1)
    parser.add_argument('--eval-batch-size', type=int, default=4)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', args.name):
        parser.error('实验名仅允许字母、数字、短横线和下划线')
    run = PROJECT / 'runs' / args.name
    data = Path(args.data_dir).resolve()
    if not (data / 'metadata.json').is_file():
        parser.error('全量编码尚未完成：需要 bpe8k-full/metadata.json')
    if not 0 < args.warmup_steps < args.steps or min(args.eval_every, args.eval_batches, args.batch_size, args.accumulation_steps, args.eval_batch_size) <= 0:
        parser.error('步数参数不合法')
    if args.resume and not (run / 'latest.pt').is_file():
        parser.error('没有可恢复的 latest.pt')
    if not args.worker:
        if not args.resume:
            try:
                run.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                parser.error('实验目录已存在；请换实验名，或使用 --resume')
        # 非阻塞锁避免同一实验同时训练。文件描述符传给后台监护进程。
        lock = (run / 'run.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('该实验已有进程运行')
        with (run / 'train.log').open('a') as log:
            command = [sys.executable, '-u', '-m', 'scripts.launch_tinystories',
                       '--name', args.name, '--worker', '--data-dir', str(data),
                       '--steps', str(args.steps), '--warmup-steps', str(args.warmup_steps),
                       '--eval-every', str(args.eval_every), '--eval-batches', str(args.eval_batches),
                       '--batch-size', str(args.batch_size), '--accumulation-steps', str(args.accumulation_steps),
                       '--eval-batch-size', str(args.eval_batch_size)]
            if args.resume:
                command.append('--resume')
            # 环境变量只传锁的文件描述符；子进程继承现有环境。
            import os
            env = dict(os.environ, TINYSTORIES_LOCK_FD=str(lock.fileno()))
            process = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True, pass_fds=(lock.fileno(),), env=env)
        (run / 'pid').write_text(str(process.pid) + '\n')
        print(f'已启动后台监护进程 PID={process.pid}\n日志：{run / "train.log"}\n状态：{run / "status.json"}')
        return

    import os
    if 'TINYSTORIES_LOCK_FD' not in os.environ:
        parser.error('--worker 为内部参数，请使用普通启动命令')
    lock_fd = int(os.environ['TINYSTORIES_LOCK_FD'])
    os.fstat(lock_fd)  # 确认继承的锁在整个监护进程期间保持打开。
    command = [sys.executable, '-u', '-m', 'scripts.train_tinystories',
        '--data-dir', str(data), '--device', 'cuda', '--steps', str(args.steps),
        '--schedule-steps', str(args.steps), '--warmup-steps', str(args.warmup_steps),
        '--lr', '0.0003', '--min-lr', '0.00003', '--batch-size', str(args.batch_size),
        '--accumulation-steps', str(args.accumulation_steps), '--seq-len', '256',
        '--eval-every', str(args.eval_every), '--eval-batches', str(args.eval_batches),
        '--eval-batch-size', str(args.eval_batch_size),
        '--keep-checkpoints', '3', '--run-dir', str(run)]
    if args.resume:
        command += ['--resume', str(run / 'latest.pt')]
    status = {'status': 'running', 'supervisor_pid': os.getpid(), 'command': command,
              'started_at': datetime.now(timezone.utc).isoformat()}

    def write_status():
        temporary = run / 'status.json.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(run / 'status.json')

    write_status()
    try:
        print('启动命令：', ' '.join(command), flush=True)
        process = subprocess.Popen(command, cwd=PROJECT, pass_fds=(lock_fd,))
        status['training_pid'] = process.pid
        write_status()
        code = process.wait()
        status.update(status='completed' if code == 0 else 'failed', exit_code=code)
    except Exception as error:
        status.update(status='failed', error=str(error))
        raise
    finally:
        status['ended_at'] = datetime.now(timezone.utc).isoformat()
        write_status()
    sys.exit(code)


if __name__ == '__main__':
    main()
