"""已提供：小模型配置，不需要先修改。全部练习可在 CPU 运行。"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    vocab_size: int = 12
    d_model: int = 16
    n_heads: int = 4
    d_ff: int = 32
    n_layers: int = 2
    max_len: int = 32
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    def __post_init__(self):
        assert self.d_model > 0 and self.n_heads > 0
        assert self.d_model % self.n_heads == 0
        assert self.d_model % 2 == 0, "位置编码练习只要求支持偶数 D"
        assert self.n_layers > 0 and self.max_len > 0
