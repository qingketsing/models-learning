"""全验证集评估与固定提示词生成，保存未经编辑的测试结果。"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from data_pipeline.tinystories import EOS, sha256
from tiny_gpt.config import GPTConfig
from tiny_gpt.model import GPT
from training.binary_data import BinaryTokenDataset
from training.trainer import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--baseline')
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    root, run = Path(args.data_dir), Path(args.run_dir)
    meta = json.loads((root/'metadata.json').read_text())
    assert sha256(root/'val.bin') == meta['splits']['val']['sha256']
    tokenizer_hash = sha256(root/'tokenizer.json')
    assert tokenizer_hash == meta['tokenizer_sha256']
    tokenizer = Tokenizer.from_file(str(root/'tokenizer.json'))
    ds = BinaryTokenDataset(root/'val.bin', 256, meta['splits']['val']['tokens'])
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'run_status': json.loads((run/'status.json').read_text()),
        'evaluation': {'blocks': len(ds), 'target_tokens': len(ds)*256,
                       'batch_size': 32, 'seq_len': 256, 'device': str(device),
                       'tokenizer_sha256': tokenizer_hash, 'val_sha256': meta['splits']['val']['sha256']},
        'checkpoints': [], 'generations': []}
    records = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    report['validation_history'] = [r for r in records if r.get('kind')=='validation']
    paths = [('best',run/'best.pt'),('latest',run/'latest.pt')]
    if args.baseline:
        paths.append(('baseline_10k',Path(args.baseline)))
    best_model = None
    for name, path in paths:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        assert ckpt['train_config']['tokenizer_sha256'] == tokenizer_hash
        model = GPT(GPTConfig(**ckpt['model_config'])).to(device)
        model.load_state_dict(ckpt['model']); model.eval()
        start = time.perf_counter()
        loss = evaluate(model,loader)
        assert math.isfinite(loss)
        row = {'name': name,'path': str(path),'step': ckpt['global_step'],
               'full_val_loss': loss,'perplexity': math.exp(loss),
               'saved_validation_state': ckpt.get('extra_state'),
               'eval_seconds': time.perf_counter()-start}
        report['checkpoints'].append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
        if name=='best': best_model = model
        del ckpt, model
    prompts = [
        ('greedy', 'Once upon a time', 42),
        ('sample', 'Once upon a time', 42),
        ('sample', 'One day, a little cat found a strange box.', 42),
        ('sample', 'Lily lost her red ball in the park. She asked her friend Tom to help her find it.', 42),
        ('sample', 'Tim broke his friend\'s toy. He wanted to tell the truth, but he was afraid.', 42),
    ]
    for mode,prompt,seed in prompts:
        torch.manual_seed(seed)
        ids = torch.tensor([tokenizer.encode(prompt,add_special_tokens=False).ids],dtype=torch.long,device=device)
        count,reason=0,'max_new_tokens'
        with torch.inference_mode():
            for _ in range(200):
                scores=best_model(ids[:,-best_model.cfg.max_len:])[:,-1,:]
                assert torch.isfinite(scores).all()
                if mode=='greedy': nxt=scores.argmax(-1,keepdim=True)
                else:
                    vals,indices=torch.topk(scores/0.8,40,dim=-1)
                    nxt=indices.gather(-1,torch.multinomial(torch.softmax(vals,-1),1))
                if nxt.item()==tokenizer.token_to_id(EOS):
                    reason='EOS';break
                ids=torch.cat([ids,nxt],dim=1);count+=1
        row={'mode':mode,'prompt':prompt,'seed':seed,'temperature':0.8 if mode=='sample' else None,
             'top_k':40 if mode=='sample' else None,'max_new_tokens':200,'new_tokens':count,
             'stop_reason':reason,'text':tokenizer.decode(ids[0].tolist())}
        report['generations'].append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['TinyStories evaluation',json.dumps(report['evaluation'],ensure_ascii=False),
           '\nCheckpoint results:',json.dumps(report['checkpoints'],ensure_ascii=False,indent=2)]
    for i,row in enumerate(report['generations'],1):
        lines.extend([f'\nTEST {i}: {row["mode"]}, seed={row["seed"]}, stop={row["stop_reason"]}, new_tokens={row["new_tokens"]}',row['text']])
    (output/'generations.txt').write_text('\n'.join(lines)+'\n')
    print('Saved:',str(output),flush=True)


if __name__=='__main__': main()
