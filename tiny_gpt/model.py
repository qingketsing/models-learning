"""组装 Tiny GPT：编号 -> Embedding -> 多层 Block -> LN -> 词表分数。

第一版只返回 logits；标签错位、交叉熵和优化器放在模型外部。
这是教学模型，尚未对齐严格 GPT-2 复现中的全部细节。
"""

from workbook_support import todo

from torch import nn

from .config import GPTConfig
from .embeddings import GPTEmbedding
from .normalization import LayerNorm
from .block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = todo('06.01：创建 GPTEmbedding')

        # ModuleList 注册每一层的参数，但不能直接当函数调用。
        # 每层独立创建；不能用 [TransformerBlock(cfg)] * cfg.n_layers。
        self.blocks = todo('06.02：ModuleList 中独立创建每一层 Block')
        self.final_norm = todo('06.03：创建最终 LayerNorm')

        # 将 D 个特征变成 V 个词表分数；第一版暂不共享 Embedding 权重。
        self.lm_head = todo('06.04：Linear(D,V,bias=False)，V 是词表大小')

    def forward(self, input_ids):
        """long [B,T] -> float logits [B,T,V]。

        logits[b,t,:] 表示读到位置 t 后，对下一个 token 的词表分数。
        输入检查由 embedding 完成；因果 mask 由每层 Attention 构建。
        """

        # 实现提示：1：调用 embedding，将编号 [B,T] 变成向量 [B,T,D]。
        x = todo('06.05：输入 embedding')

        # 实现提示：2：遍历 self.blocks，依次更新 x，每层保持 [B,T,D]。
        # 提示：for block in self.blocks: ...
        # 每层接收上一层的输出，不能每次都传最初的 embedding 输出。
        # 请在这里自己写出循环，替换下一行的省略号。
        for block in self.blocks:
            x = todo('06.06：循环内调用当前 block 更新 x')

        # 实现提示：3：调用 final_norm，得到 [B,T,D]。
        x = todo('06.07：所有 block 后再归一化')

        # 实现提示：4：调用 lm_head，得到 [B,T,V]。
        # 保留所有位置的预测，不要只取最后一个位置。
        # 返回原始分数，不做 softmax 或 argmax，以便后续计算交叉熵。
        logits = todo('06.08：lm_head 输出 [B,T,V]，不做 softmax')
        return logits
