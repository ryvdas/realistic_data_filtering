"""Tiny distributed integration check; does not load project weights or data.

Run: .venv-speedrun/bin/python -m torch.distributed.run --standalone \
    --nproc_per_node=2 tests/smoke/midtrain_optimized.py
"""
from contextlib import nullcontext
from functools import partial
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.olmo3.modeling_olmo3 import Olmo3DecoderLayer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_midtrain_optimizations import OptimizedMidtrainTest


def main():
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
    device=torch.device('cuda',local);dist.init_process_group('nccl',device_id=device)
    torch.set_num_threads(2)
    fixture=OptimizedMidtrainTest()
    for mode in ['lora','full']:
        model=fixture.model(lora=mode=='lora')
        if mode=='lora':
            wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local],find_unused_parameters=False)
        else:
            wrapped=FSDP(model,auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls={Olmo3DecoderLayer}),
                sharding_strategy=ShardingStrategy.FULL_SHARD,device_id=device,sync_module_states=True,
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.bfloat16))
        params=[p for p in wrapped.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),fused=True)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            batch=fixture.batch();batch['labels'][0,10+dist.get_rank():]=-100
            # Two repeated microbatches with unequal valid-token counts per rank.
            targets=(batch['labels'][:,1:]!=-100).sum()*2;dist.all_reduce(targets)
            for micro in range(2):
                sync=wrapped.no_sync() if mode=='lora' and micro==0 else nullcontext()
                with sync,torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=wrapped(**batch,num_items_in_batch=targets/dist.get_world_size()).loss
                    assert torch.isfinite(loss)
                    loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params if p.numel())
            norm=wrapped.clip_grad_norm_(1.) if mode=='full' else torch.nn.utils.clip_grad_norm_(params,1.)
            assert torch.isfinite(norm)
            optimizer.step()
            if mode=='lora':
                for p in params:
                    mean=p.detach().clone();dist.all_reduce(mean);mean/=dist.get_world_size()
                    torch.testing.assert_close(p,mean,rtol=0,atol=0)
        if dist.get_rank()==0:print(f'{mode}: two distributed BF16/fused optimizer steps passed',flush=True)
        del optimizer,params,wrapped,model,batch
        torch.cuda.empty_cache();dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':main()
