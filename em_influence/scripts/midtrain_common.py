"""Small CPU-only utilities for the separate continued-mid-training experiment."""
import hashlib
import json
from pathlib import Path

BASE = Path('/mnt/ssd-1/ryvaan/em_influence/midtrain_lora')
MODEL = 'allenai/Olmo-3-1025-7B'
REVISION = '373bad25002f1624757a73235c5ca844c6375c25'
CORE_COMMIT = 'ccf4ef4b5f6552e92f933589e9e6377d16fafb40'
DATA_REVISION = 'f23942ae8a8114af6e992efe8188ce8c531acd16'
TOKENIZER_REVISION = '5292e5d6c0f40b67cc765fe41bec991cf4345b5c'

def atomic_json(path, value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def implementation_sha(runner):
    """Bind preflight, resume gates and checkpoints to all imported training code."""
    runner = Path(runner)
    files = [runner, runner.with_name('midtrain_common.py'),
             runner.with_name('midtrain_optimizations.py')]
    return object_sha({p.name: sha(p) for p in files})

def batch_layout(cfg, world):
    microbatch = cfg['microbatch']; accumulation = cfg['gradient_accumulation']
    global_batch = cfg['global_batch_sequences']
    assert all(type(x) is int and x > 0 for x in [world, microbatch, accumulation, global_batch])
    assert cfg['world_size'] == world == 8
    assert global_batch == world * microbatch * accumulation, 'Global batch must equal world x microbatch x accumulation'
    return global_batch, accumulation, microbatch

def batch_indices(step, micro, rank, world, microbatch, global_batch):
    first = step * global_batch + micro * world * microbatch + rank * microbatch
    return range(first, first + microbatch)

def expected_coverage(steps, rank, world, microbatch, global_batch):
    return [index for step in range(steps)
            for micro in range(global_batch // (world * microbatch))
            for index in batch_indices(step, micro, rank, world, microbatch, global_batch)]

def training_prefix(entries, target_input_tokens, global_batch):
    """First complete optimizer window reaching the token budget; reject short data."""
    total = 0
    for end in range(global_batch, len(entries) + 1, global_batch):
        total += sum(row['input_tokens'] for row in entries[end-global_batch:end])
        if total >= target_input_tokens:
            return entries[:end], total
    raise ValueError('Prepared data has insufficient complete batches for the input-token target')

def quotas(counts,total):
    n=sum(counts.values());q={k:total*v//n for k,v in counts.items()}
    order=sorted(counts,key=lambda k:(-(total*counts[k]%n),k))
    for k in order[:total-sum(q.values())]:q[k]+=1
    assert sum(q.values())==total
    return q

class TokenLinearScheduler:
    """Linear warmup/decay using actual processed non-padding tokens, including on resume."""
    def __init__(self,optimizer,total_tokens,warmup_ratio,peak_lr):
        self.optimizer=optimizer;self.total_tokens=total_tokens;self.warmup_tokens=total_tokens*warmup_ratio
        self.peak_lr=peak_lr;self.tokens=0;self.last_epoch=0;self._apply()
    def _apply(self):
        scale=self.tokens/self.warmup_tokens if self.tokens<self.warmup_tokens else max(0.,(self.total_tokens-self.tokens)/(self.total_tokens-self.warmup_tokens))
        self.lr=self.peak_lr*scale
        for group in self.optimizer.param_groups:group['lr']=self.lr
    def step(self,tokens):
        assert tokens>=self.tokens
        self.tokens=tokens;self.last_epoch+=1;self._apply()
    def get_last_lr(self):return [self.lr for _ in self.optimizer.param_groups]
    def state_dict(self):return dict(total_tokens=self.total_tokens,warmup_tokens=self.warmup_tokens,peak_lr=self.peak_lr,tokens=self.tokens,last_epoch=self.last_epoch)
    def load_state_dict(self,state):
        assert self.total_tokens==state['total_tokens'] and self.warmup_tokens==state['warmup_tokens'] and self.peak_lr==state['peak_lr']
        self.tokens=state['tokens'];self.last_epoch=state['last_epoch'];self._apply()
