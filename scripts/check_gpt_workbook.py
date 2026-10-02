"""18 关行为检查：练习版会提示待填写；参考实现用于校验检查程序。"""
import argparse
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]


def cfg(**changes):
    from tiny_gpt.config import GPTConfig
    values = dict(vocab_size=11, max_len=12, d_model=8, n_heads=2, n_layers=2, d_ff=24)
    values.update(changes)
    return GPTConfig(**values)


def model():
    from tiny_gpt.model import GPT
    return GPT(cfg())


def check01():
    import torch
    from tiny_gpt.embeddings import GPTEmbedding
    m=GPTEmbedding(cfg());ids=torch.tensor([[1,2,3],[1,2,3]])
    out=m(ids)
    assert out.shape==(2,3,8)
    torch.testing.assert_close(out[0],m.token(ids)[0]+m.position(torch.arange(3)))
    torch.testing.assert_close(out[0],out[1])
    out.sum().backward();assert m.token.weight.grad is not None and m.position.weight.grad is not None
    try:m(torch.zeros(1,13,dtype=torch.long))
    except ValueError:pass
    else:raise AssertionError('必须拒绝超过位置表的输入')


def check02():
    import torch
    from tiny_gpt.normalization import LayerNorm
    x=torch.randn(2,3,8,requires_grad=True);m=LayerNorm(8)
    torch.testing.assert_close(m(x),torch.nn.LayerNorm(8)(x))
    assert set(dict(m.named_parameters()))=={'weight','bias'}
    assert torch.isfinite(m(torch.ones_like(x))).all()
    m(x).square().sum().backward();assert x.grad is not None


def check03():
    import torch
    from tiny_gpt.masks import causal_mask
    mask=causal_mask(4)
    assert mask.shape==(1,1,4,4) and mask.dtype==torch.bool
    assert torch.equal(mask[0,0],torch.arange(4)[None,:]>torch.arange(4)[:,None])


def check04():
    import torch
    from tiny_gpt.attention import CausalSelfAttention
    m=CausalSelfAttention(cfg());x=torch.randn(2,4,8)
    torch.testing.assert_close(m.merge_heads(m.split_heads(x)),x)
    out,w=m(x,return_weights=True)
    assert out.shape==x.shape and w.shape==(2,2,4,4)
    torch.testing.assert_close(w.sum(-1),torch.ones(2,2,4))
    assert torch.count_nonzero(torch.triu(w,diagonal=1))==0
    q=m.split_heads(m.q_proj(x));k=m.split_heads(m.k_proj(x));v=m.split_heads(m.v_proj(x))
    scores=(q@k.transpose(-2,-1)/math.sqrt(m.head_dim)).masked_fill(torch.ones(4,4,dtype=torch.bool).triu(1),float('-inf'))
    expected=m.out_proj(m.merge_heads(scores.softmax(-1)@v))
    torch.testing.assert_close(out,expected)
    other=x.clone();other[:,2:]+=10
    torch.testing.assert_close(m(x)[:,:2],m(other)[:,:2])


def check05():
    import torch
    from tiny_gpt.mlp import MLP
    from tiny_gpt.block import TransformerBlock
    x=torch.randn(2,4,8);mlp=MLP(cfg())
    torch.testing.assert_close(mlp(x),mlp.down(torch.nn.functional.gelu(mlp.up(x),approximate='tanh')))
    block=TransformerBlock(cfg());r=x+block.attention(block.norm1(x))
    torch.testing.assert_close(block(x),r+block.mlp(block.norm2(r)))


def check06():
    import torch
    m=model();ids=torch.tensor([[1,2,3,4],[4,3,2,1]])
    out=m(ids);assert out.shape==(2,4,11)
    assert m.blocks[0] is not m.blocks[1]
    other=ids.clone();other[:,2:]=5
    torch.testing.assert_close(out[:,:2],m(other)[:,:2])
    out.square().mean().backward()
    assert all(p.grad is not None for p in m.parameters())


def check07():
    from training.data import TokenDataset
    ds=TokenDataset([1,2,3,4,5,6],3)
    assert len(ds)==3 and ds[0][0].tolist()==[1,2,3] and ds[0][1].tolist()==[2,3,4]
    assert ds[2][1].tolist()==[4,5,6]
    try:ds[3]
    except IndexError:pass
    else:raise AssertionError('越界需抛错')


def check08():
    import torch
    from training.trainer import train_step
    m=model();x=torch.tensor([[1,2,3]]);y=torch.tensor([[2,3,4]])
    expected=torch.nn.functional.cross_entropy(m(x).reshape(-1,11),y.reshape(-1)).item()
    before=[p.detach().clone() for p in m.parameters()]
    got=train_step(m,torch.optim.SGD(m.parameters(),lr=.1),x,y)
    assert abs(got-expected)<1e-6
    assert any(not torch.equal(a,b) for a,b in zip(before,m.parameters()))


def check09():
    import torch
    from training.trainer import evaluate
    m=model();m.train();before={k:v.clone() for k,v in m.state_dict().items()}
    data=[(torch.tensor([[1,2]]),torch.tensor([[2,3]])),(torch.tensor([[3,4,5]]),torch.tensor([[4,5,6]]))]
    expected=sum(torch.nn.functional.cross_entropy(m(x).reshape(-1,11),y.reshape(-1)).item()*y.numel() for x,y in data)/5
    assert abs(evaluate(m,data)-expected)<1e-6 and m.training
    for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    assert all(p.grad is None for p in m.parameters())
    try:evaluate(m,[])
    except ValueError:pass
    else:raise AssertionError('空验证集应抛错')
    assert m.training


def check10():
    import copy,torch
    from training.trainer import train_step,train_accumulated_step,clip_gradients
    a=model();b=copy.deepcopy(a)
    x=torch.tensor([[1,2,3],[2,3,4],[3,4,5]]);y=x+1
    train_step(a,torch.optim.SGD(a.parameters(),lr=.1),x,y)
    train_accumulated_step(b,torch.optim.SGD(b.parameters(),lr=.1),[(x[:2],y[:2]),(x[2:],y[2:])])
    for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa,pb,rtol=1e-5,atol=1e-6)
    for p in b.parameters():p.grad=torch.ones_like(p)*10
    clip_gradients(b,.5)
    norm=torch.sqrt(sum(p.grad.square().sum() for p in b.parameters()))
    assert norm<=.50001


def check11():
    import torch
    from training.checkpoint import save_checkpoint,load_checkpoint
    from training.trainer import train_step
    a=model();opt=torch.optim.Adam(a.parameters());x=torch.tensor([[1,2]]);y=x+1
    train_step(a,opt,x,y)
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'model.pt';save_checkpoint(path,a,opt,2,7,{'data':'toy'})
        b=model();other=torch.optim.Adam(b.parameters())
        assert load_checkpoint(path,b,other,{'data':'toy'})==(2,7)
        torch.testing.assert_close(a(x),b(x));assert len(other.state)==len(opt.state)>0
        try:load_checkpoint(path,b,other,{'data':'wrong'})
        except ValueError:pass
        else:raise AssertionError('应拒绝不同训练配置')


def check12():
    import torch
    from training.schedule import get_lr,set_lr
    kw=dict(base_lr=.01,min_lr=.001,warmup_steps=10,total_steps=150)
    for step,value in [(1,.001),(10,.01),(80,.0055),(150,.001),(160,.001)]:
        assert abs(get_lr(step,**kw)-value)<1e-12
    opt=torch.optim.SGD(model().parameters(),lr=.1);set_lr(opt,.02);assert opt.param_groups[0]['lr']==.02


def check13():
    from data_pipeline.tokenizer import CharTokenizer
    a=CharTokenizer.from_text('我 爱你\n我');b=CharTokenizer.from_text('你我\n 爱')
    assert a.id_to_token==b.id_to_token
    text='我 爱你\n';assert a.decode(a.encode(text))==text
    assert a.encode('🦊')==[0] and a.decode([0])=='<unk>'
    assert a.encode('')==[] and a.decode([])==''
    with tempfile.TemporaryDirectory() as d:
        a.save(Path(d)/'vocab.json');assert CharTokenizer.load(Path(d)/'vocab.json').encode(text)==a.encode(text)
    try:a.decode([a.vocab_size])
    except ValueError:pass
    else:raise AssertionError('非法编号应抛错')


def check14():
    import torch
    from training.device import resolve_device,move_batch,model_device
    assert resolve_device('cpu')==torch.device('cpu')
    x=torch.ones(1,2,dtype=torch.long);a,b=move_batch(x,x,torch.device('cpu'))
    assert a.dtype==b.dtype==torch.long and model_device(model()).type=='cpu'


def check15():
    from data_pipeline.tinystories import iter_stories,train_bpe,EOS
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'stories.txt';path.write_text(f'First story.\nSecond line.\n{EOS}\n\n{EOS}\nLast story.',encoding='utf-8')
        assert list(iter_stories(path))==['First story.\nSecond line.','Last story.']
        assert len(list(iter_stories(path,1)))==1
        tokenizer,count=train_bpe(path,2,300);assert count==2
        for text in ['New text.\n','中文 🦊 café\t']:
            assert tokenizer.decode(tokenizer.encode(text).ids)==text


def check16():
    from scripts.check_tinystories import main
    main()


def check17():
    from scripts.check_long_training import main
    main()


def check18():
    import torch
    from data_pipeline.tinystories import train_bpe,sha256
    from tiny_gpt.model import GPT
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);(root/'stories.txt').write_text('A cat smiled. A dog smiled.',encoding='utf-8')
        tok,_=train_bpe(root/'stories.txt',1,300);tok.save(str(root/'tokenizer.json'))
        m=GPT(cfg(vocab_size=tok.get_vocab_size()))
        with torch.no_grad():m.lm_head.weight.zero_()  # EOS=0，贪心应立即停止
        torch.save({'model_config':vars(m.cfg),'model':m.state_dict(),'global_step':0,
                    'train_config':{'tokenizer_sha256':sha256(root/'tokenizer.json')}},root/'model.pt')
        cmd=[sys.executable,'-m','scripts.generate_tinystories','--checkpoint',str(root/'model.pt'),
             '--tokenizer',str(root/'tokenizer.json'),'--prompt','A cat','--device','cpu','--max-new-tokens','3']
        result=subprocess.run(cmd,cwd=IMPLEMENTATION,text=True,capture_output=True)
        if result.returncode:
            if 'NotImplementedError: TODO' in result.stderr:raise NotImplementedError(result.stderr.splitlines()[-1])
            raise AssertionError(result.stderr)
        assert '新增 0 个非 EOS token' in result.stdout and 'EOS' in result.stdout
        result=subprocess.run(cmd+['--sample'],cwd=IMPLEMENTATION,text=True,capture_output=True)
        if result.returncode:
            if 'NotImplementedError: TODO' in result.stderr:raise NotImplementedError(result.stderr.splitlines()[-1])
            raise AssertionError(result.stderr)
        assert '生成结果' in result.stdout


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=[f'{i:02d}' for i in range(1,19)]+['all'])
    p.add_argument('--reference',action='store_true',help='仅用于检查完整参考实现')
    p.add_argument('--trace',action='store_true')
    args=p.parse_args()
    IMPLEMENTATION=ROOT/'reference_impl' if args.reference else ROOT
    sys.path.insert(0,str(IMPLEMENTATION))
    import torch
    torch.manual_seed(42);torch.set_num_threads(1)
    failures,pending=0,0
    stages=range(1,19) if args.stage=='all' else [int(args.stage)]
    for stage in stages:
        try:
            globals()[f'check{stage:02d}']()
            print(f'[通过] {stage:02d}')
        except NotImplementedError as error:
            pending+=1;print(f'[待填写] {stage:02d}: {error}')
        except Exception as error:
            failures+=1;print(f'[未通过] {stage:02d}: {type(error).__name__}: {error}')
            if args.trace:traceback.print_exc()
    if failures or pending:sys.exit(1 if failures else 2)
