# 完整参考实现

这里保存创建 TODO 练习之前的完整 GPT、训练系统、tokenizer、数据处理和运行脚本。根目录对应代码是练习版；此目录是独立可运行的参考项目。对照时只看当前函数，理解后回到根目录自己填写。

## 环境与检查

先在仓库根目录准备 Python 环境：

```bash
uv sync --locked
uv pip install --python .venv/bin/python -r requirements-data.txt
source .venv/bin/activate
python scripts/check_gpt_workbook.py all --reference
```

测试使用同一个检查程序，避免练习与答案采用不同验收标准。

## 独立运行

完整参考文件保留原来的包名与绝对包导入。必须先进入本目录，再使用仓库根目录的虚拟环境解释器，否则可能导入根目录的 TODO 版本。

```bash
cd reference_impl
../.venv/bin/python -m scripts.train --device cpu --epochs 2 --checkpoint checkpoints/reference-toy.pt
```

`examples/corpus/` 已随参考项目复制。运行输出属于此目录，忽略规则覆盖这些文件。

真实训练入口仍需外部原始数据、tokenizer、二进制数据和 checkpoint。它们没有随代码提交。历史默认服务器路径需要按你的机器调整；生成时显式传 `--checkpoint` 和 `--tokenizer`，不要把示例目录当作已经存在的模型。

本目录保存的是完整实现快照，不会随你填写练习自动同步。不要在 GPU 节点用本机 CPU PyTorch 源覆盖已有 CUDA 环境。
