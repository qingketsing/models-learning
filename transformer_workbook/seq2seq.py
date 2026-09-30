"""第 09 关：补齐 Encoder–Decoder，不与主线的 decoder-only 混淆。
采用统一 Pre-Norm，非 2017 原始 Post-Norm 的逐参数复刻。
"""
from torch import nn
from .attention import MultiHeadAttention
from .normalization import LayerNorm
from .blocks import FeedForward
from .embeddings import TokenPositionEmbedding
from .masks import padding_mask, decoder_mask


class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads)
        self.norm2 = LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff)

    def forward(self, x, source_mask):
        """TODO 09a：与 DecoderBlock 类似，但只有 source padding mask，
        没有 causal mask；源文本的有效位置可以双向关注。
        """
        raise NotImplementedError("TODO 09a：Encoder Block")


class TranslationDecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg.d_model)
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads)
        self.norm2 = LayerNorm(cfg.d_model)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads)
        self.norm3 = LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff)

    def forward(self, x, memory, target_mask, source_mask):
        """TODO 09b：三次 Pre-Norm 残差。
        1. target self-attention：Q/K/V 都来自 norm1(x)，target_mask。
        2. cross-attention：Q 来自 norm2(更新后的 x)，K/V 来自 memory，
           mask 使用 source_mask；不要把 source 的 T 写成 target 的 T。
        3. FFN(norm3(再次更新后的 x))，加回第二步结果。
        memory 是整个 Encoder 的输出 [B,S,D]，本层不再归一化它。
        """
        raise NotImplementedError("TODO 09b：带交叉注意力的 Decoder Block")


class TinySeq2Seq(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        args = (cfg.vocab_size, cfg.d_model, cfg.max_len, cfg.pad_id)
        self.source_embedding = TokenPositionEmbedding(*args)
        self.target_embedding = TokenPositionEmbedding(*args)
        self.encoders = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.decoders = nn.ModuleList([TranslationDecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.encoder_norm = LayerNorm(cfg.d_model)
        self.decoder_norm = LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, source, target_input):
        """source [B,S]，target_input [B,T] -> logits [B,T,V]。
        TODO 09c：source_embedding -> encoders -> encoder_norm 得到 memory；
        target_embedding -> decoders(传入 memory) -> decoder_norm -> lm_head。
        source padding mask 同时供 Encoder self-attn 与 Decoder cross-attn 用。
        target_input 已在外部错位，forward 内不要再次 shift。
        """
        if source.eq(self.cfg.pad_id).all(dim=-1).any():
            raise ValueError("源序列不能全是 PAD")
        raise NotImplementedError("TODO 09c：完整 Encoder–Decoder")
