# tiny_gpt/config.py

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 10 # token表的行数
    max_len: int = 16 # 允许输入的最大长度
    d_model: int = 8 # 每个block的向量维度
    n_heads: int = 2 # 多头注意力机制的头数
    n_layers: int = 2 # Block的层数
    d_ff: int = 32 # MLP中间维度

    def __post_init__(self):
        assert self.vocab_size > 0
        assert self.max_len > 0
        assert self.d_model > 0
        assert self.n_heads > 0
        assert self.d_model % self.n_heads == 0
        assert self.n_layers > 0
        assert self.d_ff > 0