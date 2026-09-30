"""验收入口：python -m transformer_workbook.check 01（或 all）。无需修改。"""
import argparse
import math
import traceback
import torch
import torch.nn.functional as F
from torch import nn
from .config import Config
from . import embeddings, normalization, masks, attention, blocks, model, objective, training, generation, seq2seq
from .data import language_batch


def close(a, b, message="数值不一致"):
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6, msg=message)


def rejects(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("此非法输入应该抛出 ValueError")


def check01():
    p = embeddings.sinusoidal_table(7, 6)
    assert p.shape == (1, 7, 6) and p.dtype == torch.float32 and not p.requires_grad
    for pos in (0, 1, 5):
        expected = [f(pos / 10000 ** (2 * i / 6)) for i in range(3) for f in (math.sin, math.cos)]
        close(p[0, pos], torch.tensor(expected), "检查 sin/cos 列与频率指数")
    m = embeddings.TokenPositionEmbedding(12, 6, 7)
    ids = torch.tensor([[1, 3, 4], [1, 3, 5]])
    out = m(ids)
    close(out - m.token(ids), p[:, :3].expand(2, -1, -1))
    assert 'position' in dict(m.named_buffers()) and 'position' not in dict(m.named_parameters())
    out.sum().backward()
    assert m.token.weight.grad is not None
    rejects(lambda: m(torch.ones(1, 8, dtype=torch.long)))


def check02():
    m = normalization.LayerNorm(6).double()
    with torch.no_grad():
        m.weight.copy_(torch.linspace(0.5, 1.5, 6)); m.bias.copy_(torch.linspace(-1, 1, 6))
    x = torch.randn(2, 3, 6, dtype=torch.double, requires_grad=True)
    expected = F.layer_norm(x, (6,), m.weight, m.bias, m.eps)
    close(m(x), expected, "LN 应在最后一轴，用总体方差和可训练仿射参数")
    a = torch.autograd.grad(m(x).square().sum(), (x, m.weight, m.bias))
    b = torch.autograd.grad(expected.square().sum(), (x, m.weight, m.bias))
    for actual, wanted in zip(a, b): close(actual, wanted, "LayerNorm 梯度不一致")
    assert torch.isfinite(m(torch.ones_like(x))).all(), "常量输入也不能 NaN"


def check03():
    c = masks.causal_mask(3)
    assert c.dtype == torch.bool and c.shape == (1, 1, 3, 3)
    assert torch.equal(c[0, 0], torch.tensor([[False, True, True], [False, False, True], [False, False, False]]))
    ids = torch.tensor([[1, 3, 0], [1, 4, 5]])
    p = masks.padding_mask(ids)
    assert p.shape == (2, 1, 1, 3) and p.dtype == torch.bool
    assert p[0, 0, 0, 2] and not p[1].any()
    d = masks.decoder_mask(ids)
    close(d, c | p)
    rejects(lambda: masks.decoder_mask(torch.tensor([[0, 1]])))
    rejects(lambda: masks.decoder_mask(torch.tensor([[1, 0, 2]])))


def check04():
    q = torch.randn(2, 2, 3, 4, requires_grad=True)
    k = torch.randn(2, 2, 5, 4, requires_grad=True)
    v = torch.randn(2, 2, 5, 4, requires_grad=True)
    blocked = torch.zeros(2, 1, 1, 5, dtype=torch.bool); blocked[..., -1] = True
    out, weights = attention.scaled_attention(q, k, v, blocked)
    assert weights.shape == (2, 2, 3, 5) and out.shape == q.shape
    close(weights.sum(-1), torch.ones(2, 2, 3))
    assert torch.count_nonzero(weights[..., -1]) == 0
    # SDPA 的 True 表示允许，本练习的 True 表示屏蔽，所以必须取反。
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=~blocked, dropout_p=0.0)
    close(out, ref, "检查缩放、softmax 轴、mask 方向")
    ga = torch.autograd.grad(out.square().sum(), (q, k, v))
    gb = torch.autograd.grad(ref.square().sum(), (q, k, v))
    for a, b in zip(ga, gb): close(a, b, "注意力梯度不一致")
    rejects(lambda: attention.scaled_attention(q, k, v, torch.ones_like(blocked)))
    m = attention.MultiHeadAttention(8, 2)
    x = torch.arange(48.).reshape(2, 3, 8)
    split = m.split_heads(x)
    close(split[:, 1, :, :], x[:, :, 4:], "拆头后元素对应错误")
    close(m.merge_heads(split), x)
    query, memory = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
    output, _ = m(query, memory, memory, blocked)
    reference = nn.MultiheadAttention(8, 2, bias=False, batch_first=True)
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat([m.q_proj.weight, m.k_proj.weight, m.v_proj.weight]))
        reference.out_proj.weight.copy_(m.out_proj.weight)
    expected, _ = reference(query, memory, memory, key_padding_mask=blocked[:, 0, 0])
    close(output, expected, "多头注意力与独立实现不一致；检查 W_o 与不同的 Tq/Tk")


def check05():
    cfg = Config(d_model=8, n_heads=2, d_ff=12, n_layers=1)
    f = blocks.FeedForward(8, 12)
    x = torch.randn(2, 3, 8, requires_grad=True)
    close(f(x), f.down(F.gelu(f.up(x), approximate='tanh')))
    block = blocks.DecoderBlock(cfg)
    mask = masks.causal_mask(3)
    u = block.norm1(x)
    r = x + block.attn(u, u, u, mask)[0]
    expected = r + block.ffn(block.norm2(r))
    out = block(x, mask)
    close(out, expected, "检查 Pre-Norm 顺序、二次残差基底")
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
    with torch.no_grad():
        for sub in (block.attn, block.ffn):
            for p in sub.parameters(): p.zero_()
    close(block(x.detach(), mask), x.detach(), "子层归零后，残差应保留原输入")


def check06():
    cfg = Config()
    m = model.TinyLanguageModel(cfg).eval()
    ids = torch.tensor([[1, 3, 4, 5], [1, 7, 8, 9]])
    out = m(ids)
    assert out.shape == (2, 4, cfg.vocab_size) and torch.isfinite(out).all()
    h = m.embedding(ids)
    blocked = masks.decoder_mask(ids)
    for block in m.blocks: h = block(h, blocked)
    close(out, m.lm_head(m.final_norm(h)), "检查模块串联与输出原始 logits")
    changed = ids.clone(); changed[:, -1] = 10
    close(out[:, :-1], m(changed)[:, :-1], "修改未来 token 不应改变前缀 logits")
    close(out, m(torch.cat([ids, torch.zeros(2, 2, dtype=torch.long)], 1))[:, :4], "追加右 PAD 不应改变原位置")
    assert m.blocks[0].attn.q_proj.weight is not m.blocks[1].attn.q_proj.weight
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def check07():
    tokens = torch.tensor([[1, 3, 4, 2, 0], [1, 7, 2, 0, 0]])
    x, y = objective.shift_tokens(tokens)
    assert torch.equal(x, tokens[:, :-1]) and torch.equal(y, tokens[:, 1:])
    z = torch.randn(2, 4, 12, requires_grad=True)
    loss = objective.token_loss(z, y)
    keep = y.ne(0)
    close(loss, F.cross_entropy(z[keep], y[keep]))
    assert loss.shape == torch.Size([])
    loss.backward()
    assert torch.count_nonzero(z.grad[~keep]) == 0, "PAD 标签必须不产生损失梯度"
    altered = z.detach().clone(); altered[~keep] = 100 * torch.randn_like(altered[~keep])
    close(loss.detach(), objective.token_loss(altered, y), "PAD 位置的 logits 不应影响 loss")
    rejects(lambda: objective.token_loss(z, torch.zeros_like(y)))


def check08():
    # 对照单步 SGD，故意预填旧梯度，检查有没有 zero_grad。
    import copy
    cfg = Config(n_layers=1)
    m = model.TinyLanguageModel(cfg)
    reference = copy.deepcopy(m)
    x, y = objective.shift_tokens(language_batch())
    optimizer = torch.optim.SGD(m.parameters(), lr=0.03)
    opt_ref = torch.optim.SGD(reference.parameters(), lr=0.03)
    for p in m.parameters(): p.grad = torch.ones_like(p)
    m.eval()
    actual = training.train_step(m, optimizer, x, y)
    opt_ref.zero_grad(); expected = objective.token_loss(reference(x), y)
    expected.backward(); opt_ref.step()
    assert isinstance(actual, float) and m.training
    assert abs(actual - expected.item()) < 1e-5
    for a, b in zip(m.parameters(), reference.parameters()): close(a, b, "单步更新不一致；检查梯度清空/顺序")
    # 合成模型让末位置有确定答案，验证 EOS、模式恢复与 no_grad。
    class Scripted(nn.Module):
        def __init__(self):
            super().__init__(); self.cfg = Config(max_len=6); self.calls = 0
        def forward(self, ids):
            assert not self.training and not torch.is_grad_enabled()
            self.calls += 1
            z = torch.zeros(1, ids.size(1), 12)
            z[:, :, 8] = 5
            z[:, -1, :] = 0
            z[:, -1, 3 if ids.size(1) == 1 else 2] = 10
            return z
    scripted = Scripted()
    prompt = torch.tensor([[1]])
    result = generation.greedy_generate(scripted, prompt, 5)
    assert torch.equal(result, torch.tensor([[1, 3, 2]])) and scripted.calls == 2
    assert scripted.training and torch.equal(prompt, torch.tensor([[1]]))
    scripted.eval()
    close(generation.greedy_generate(scripted, prompt, 0), prompt)
    assert not scripted.training
    close(generation.greedy_generate(scripted, torch.tensor([[1, 2]]), 3), torch.tensor([[1, 2]]))
    scripted.cfg = Config(max_len=1)
    close(generation.greedy_generate(scripted, prompt, 3), prompt)


def check09():
    cfg = Config(d_model=8, n_heads=2, d_ff=16, n_layers=1)
    m = seq2seq.TinySeq2Seq(cfg).eval()
    src = torch.tensor([[3, 4, 5, 2, 0], [7, 8, 2, 0, 0]])
    tgt = torch.tensor([[1, 3, 4], [1, 7, 8]])
    out = m(src, tgt)
    assert out.shape == (2, 3, 12)
    sm = masks.padding_mask(src); tm = masks.decoder_mask(tgt)
    memory = m.source_embedding(src)
    for enc in m.encoders:
        u = enc.norm1(memory); r = memory + enc.attn(u, u, u, sm)[0]
        expected = r + enc.ffn(enc.norm2(r))
        close(enc(memory, sm), expected, "Encoder 残差错误")
        memory = expected
    memory = m.encoder_norm(memory)
    h = m.target_embedding(tgt)
    for dec in m.decoders:
        u = dec.norm1(h); r = h + dec.self_attn(u, u, u, tm)[0]
        s = r + dec.cross_attn(dec.norm2(r), memory, memory, sm)[0]
        expected = s + dec.ffn(dec.norm3(s))
        close(dec(h, memory, tm, sm), expected, "Decoder cross-attention 的来源/残差错误")
        h = expected
    close(out, m.lm_head(m.decoder_norm(h)))
    changed = tgt.clone(); changed[:, -1] = 9
    close(out[:, :-1], m(src, changed)[:, :-1], "Decoder 偷看未来")
    longer = torch.cat([src, torch.zeros(2, 2, dtype=torch.long)], 1)
    close(out, m(longer, tgt), "源序列 padding 未正确屏蔽")
    changed_src = src.clone(); changed_src[0, 0] = 9
    assert not torch.allclose(out[0], m(changed_src, tgt)[0]), "输出应依赖 Encoder memory"
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


CHECKS = {f'{i:02}': globals()[f'check{i:02}'] for i in range(1, 10)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=[*CHECKS, 'all'])
    parser.add_argument('--trace', action='store_true', help='显示完整错误位置')
    args = parser.parse_args()
    torch.set_num_threads(1)
    for stage in (list(CHECKS) if args.stage == 'all' else [args.stage]):
        torch.manual_seed(2026)
        try:
            CHECKS[stage]()
        except NotImplementedError as error:
            print(f'[待填写] 第 {stage} 关：{error}。见 HINTS.md；后续关卡依赖前面的实现。')
            raise SystemExit(2)
        except Exception as error:
            print(f'[未通过] 第 {stage} 关：{type(error).__name__}: {error}')
            if args.trace: traceback.print_exc()
            else: print('添加 --trace 查看出错行；不要修改检查程序来绕过失败。')
            raise SystemExit(1)
        print(f'[通过] 第 {stage} 关：本关数值与行为检查通过。请再用自己的话解释。')


if __name__ == '__main__':
    main()
