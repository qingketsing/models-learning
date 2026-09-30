# 开始前：从你写过的 w、b 到模型类

你已经会 `x @ w + b`、梯度和优化器。这次并不是换了一套数学，而是把重复使用的计算封装成模块。

## 先认识这段小例子（它不是 Transformer 答案）

```python
import torch
from torch import nn

class TwoFeatureClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 3)

    def forward(self, x):
        return self.projection(x)

model = TwoFeatureClassifier()
x = torch.randn(4, 2)
print(model(x).shape)  # [4,3]
print(model.projection.weight.shape)  # [3,2]
```

- `class` 是定义一种对象；`self` 是当前这个模型对象。
- `__init__` 在创建模型时执行一次。层和参数要在这里创建。
- `super().__init__()` 初始化 PyTorch 的模块管理能力，不要省略。
- `forward` 定义收到输入后怎样计算。`model(x)` 会通过模块机制调用它。
- `nn.Linear(2,3)` 有 2 个输入特征、3 个输出分数，内部计算 `x @ weight.T + bias`。
- `nn.Linear` 的 weight 为 `[输出,输入]`，与你手写的 `[输入,输出]` 矩阵恰好转置。
- `model.parameters()` 能找出注册在这个模型及其子模块里的所有参数。

## 三种需要注册的东西

| 类型 | 用途 | 本练习的例子 |
|---|---|---|
| `nn.Parameter` | 要被优化器更新的 Tensor | LayerNorm 的 weight、bias |
| `nn.ModuleList` | 注册多个子模块 | 多层 DecoderBlock |
| `register_buffer` | 不是训练参数、但要随模型移动和保存的 Tensor | 正弦位置编码表 |

普通 Python 列表中装的层不自动注册；这里已帮你写好 ModuleList。

## Embedding 是查表，不是整数乘权重

```python
table = nn.Embedding(10, 6)
ids = torch.tensor([[2, 4, 2]], dtype=torch.long)
x = table(ids)
print(x.shape)  # [1,3,6]
```

意思是从 `[10,6]` 的参数表中读取第 2、4、2 行。同一个 ID 在添加位置编码之前，查到的向量相同。Embedding 表通过梯度更新，token ID 本身不求梯度。

## TODO 的使用方法

文件中的 `raise NotImplementedError("TODO ...")` 是刻意保留的断点。读完注释后，把它替换成你的实现和 return；不要只删除它，也不要全部改成 pass。

例如一个简单练习要求 `[B,T,D] -> [B,D]`，沿 token 轴平均：

```python
# 原来
# raise NotImplementedError("求 token 平均")

# 你填写的局部实现
# return x.mean(dim=1)
```

第一次只打开 `embeddings.py`，做 01a、01b。给自己纸上画出 `[B,T] -> [B,T,D]` 的变化，再运行第 01 关检查。
