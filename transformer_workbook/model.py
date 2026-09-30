"""第 06 关：完整 Decoder-only 语言模型。"""
from torch import nn
from .embeddings import TokenPositionEmbedding
from .normalization import LayerNorm
from .blocks import DecoderBlock
from .masks import decoder_mask


class TinyLanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = TokenPositionEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.max_len, cfg.pad_id)
        # 每层独立创建；不要用 [同一个block] * n_layers 来重复引用。
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, ids):
        """long [B,T] -> logits [B,T,V]。
        TODO 06：构建 decoder_mask；embedding；依次经过 blocks；
        final_norm；lm_head。返回 logits，不在这里 softmax。
        词表投影的 V 不等于 D，也不等于序列长度 T。
        """
        raise NotImplementedError("TODO 06：连接完整语言模型")
