"""Eight-rank FSDP full-parameter causal continued-training benchmark with atomic, exact-position resume."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import threading
import time
from contextlib import nullcontext
import numpy as np
import torch
import torch.distributed as dist
from functools import partial
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType, LocalStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.olmo3.modeling_olmo3 import Olmo3DecoderLayer
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM,AutoTokenizer,get_scheduler,set_seed
from midtrain_common import atomic_json,sha,object_sha,BASE,TokenLinearScheduler,batch_layout,batch_indices,expected_coverage,implementation_sha
from midtrain_optimizations import configure_model


def gpu_inventory():
    value=os.popen('nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits').read()
    return [[int(x.strip()) for x in l.split(',')] for l in value.splitlines()]

def digest(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if p.requires_grad:h.update(n.encode());h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()

def all_memory(device):
    item=dict(rank=dist.get_rank(),allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        total=torch.cuda.get_device_properties(device).total_memory)
    rows=[None]*dist.get_world_size();dist.all_gather_object(rows,item);return rows

def batch_from_tokens(tokens,device,pad):
    inputs=torch.as_tensor(np.array(tokens,dtype=np.int64,copy=True),device=device)
    if inputs.ndim==1:inputs=inputs.unsqueeze(0)
    labels=inputs.clone();labels[inputs==pad]=-100;labels[:,0]=-100
    return dict(input_ids=inputs,attention_mask=torch.ones_like(inputs),labels=labels)

def main(a):
    launched=time.monotonic();cfg=json.loads(a.config.read_text());config_hash=sha(a.config)
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    global_batch,accumulation,microbatch=batch_layout(cfg,world)
    if a.mode in ['pilot','full']:
        gate=json.loads(a.preflight_report.read_text());resume_gate=json.loads(a.resume_verification.read_text())
        for report in [gate,resume_gate]:
            assert report['passed'] and report['config_sha256']==config_hash and report['runner_sha256']==implementation_sha(Path(__file__))
    assert cfg['model_commit']=='373bad25002f1624757a73235c5ca844c6375c25'
    if rank==0:
        lock=(BASE.parent/'speedrun_training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inventory=gpu_inventory();assert len(inventory)==8 and all(u==0 and m<256 for _,u,m,_ in inventory),inventory
        assert not a.output.exists(),'Never overwrite an existing run'
        a.output.mkdir(parents=True)
        atomic_json(a.output/'launch_ready.json',dict(idle_verified_before_cuda_initialization=True))
    else:
        deadline=time.monotonic()+120
        while not (a.output/'launch_ready.json').exists():
            assert time.monotonic()<deadline,'Rank 0 did not finish the pre-CUDA idle gate'
            time.sleep(.1)
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',device_id=device);dist.barrier();set_seed(cfg['seed'])
    assignments=[None]*world;dist.all_gather_object(assignments,dict(rank=rank,local_rank=local,uuid=str(torch.cuda.get_device_properties(local).uuid)))
    assert len({x['uuid'] for x in assignments})==8 and {x['rank'] for x in assignments}==set(range(8))
    cache=Path(cfg.get('dataset_dir',BASE/'full'));manifest=json.loads((cache/'manifest.json').read_text())
    manifest_hash=sha(cache/'manifest.json');assert manifest['tokens_sha256']==sha(cache/'tokens.u32')
    data=np.memmap(cache/'tokens.u32',dtype='<u4',mode='r',shape=(manifest['sequences'],cfg['sequence_length']))
    assert len({r['id'] for r in manifest['entries']})==len(data)
    stop=a.stop_after_step or (len(data)//global_batch if a.mode=='full' else 3 if a.mode=='preflight' else cfg['pilot_steps'])
    assert stop*global_batch<=len(data)
    tokenizer_path=snapshot_download(cfg['tokenizer_model'],revision=cfg['tokenizer_commit'],local_files_only=True)
    tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    base_tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['model_commit'],local_files_only=True)
    assert tok.get_vocab()==base_tok.get_vocab(), 'Pretokenized IDs must match the approved base'
    assert tok.eos_token_id==100257 and tok.pad_token_id==100277
    model=AutoModelForCausalLM.from_pretrained(cfg['model'],revision=cfg['model_commit'],dtype=torch.float32,local_files_only=True,attn_implementation=cfg.get('attention_backend','flash_attention_2'))
    configure_model(model,cfg)
    model.requires_grad_(True);model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    trainable=sum(p.numel() for p in model.parameters());frozen=0
    assert trainable==7298011136 and all(p.requires_grad for p in model.parameters())
    wrapped=FSDP(model,auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls={Olmo3DecoderLayer}),
        sharding_strategy=ShardingStrategy.FULL_SHARD,device_id=device,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.bfloat16),
        limit_all_gathers=True,sync_module_states=True)
    params=list(wrapped.parameters())
    optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],betas=tuple(cfg['betas']),eps=cfg['epsilon'],weight_decay=cfg['weight_decay'],fused=cfg.get('fused_adamw',True))
    scheduler=TokenLinearScheduler(optimizer,cfg['target_input_tokens'],cfg['warmup_ratio'],cfg['learning_rate'])
    step=0;input_count=0;loss_count=0;seen=[];timed_input=0;timed_loss=0;timed_seconds=0;checkpoint_times=[]
    if a.resume:
        marker=json.loads((a.resume/'complete.json').read_text());assert marker['complete']
        assert marker['runner_sha256']==implementation_sha(Path(__file__)), 'Training implementation changed; use a fresh run'
        assert marker['config_sha256']==config_hash and marker['manifest_sha256']==manifest_hash and marker['mode']==a.mode
        for name,expected in marker['files_sha256'].items():assert sha(a.resume/name)==expected
        state=torch.load(a.resume/f'rank-{rank}.pt',map_location='cpu',weights_only=False)
        with FSDP.state_dict_type(wrapped,StateDictType.LOCAL_STATE_DICT,LocalStateDictConfig(offload_to_cpu=False)):
            wrapped.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        assert digest(wrapped)==state['parameter_digest']
        for pid,item in state['optimizer']['state'].items():
            for key,value in item.items():
                actual=optimizer.state_dict()['state'][pid][key]
                if isinstance(value,torch.Tensor):assert torch.equal(value.cpu(),actual.cpu())
                else:assert value==actual
        step=state['step'];input_count=state['input_tokens'];loss_count=state['loss_tokens']
        timed_input=state['timed_input'];timed_loss=state['timed_loss'];timed_seconds=state['timed_seconds']
        assert state['next_global_sequence']==step*global_batch and scheduler.last_epoch==step
        rng=state['rng']
        seen=rng['seen'];assert seen==expected_coverage(step,rank,world,microbatch,global_batch)
        assert input_count==sum(r['input_tokens'] for r in manifest['entries'][:step*global_batch])
        assert loss_count==sum(r['loss_tokens'] for r in manifest['entries'][:step*global_batch]) and scheduler.tokens==input_count
        random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'],device)
        assert torch.equal(torch.get_rng_state(),rng['torch']) and torch.equal(torch.cuda.get_rng_state(device),rng['cuda'])
        atomic_json(a.output/f'resume_rank_{rank}.json',dict(restored_step=step,coverage_prefix_verified=True,rng_verified=True,
            optimizer_restored=True,optimizer_exactly_restored=True,scheduler_restored=True,model_exactly_restored=True,manifest_sha256=manifest_hash))
        del state,rng
    atomic_json(a.output/f'rank_{rank}.json',dict(rank=rank,local_rank=local,world=world,trainable_parameters=trainable,frozen_parameters=frozen,
        all_parameters_trainable=True,config_sha256=config_hash,manifest_sha256=manifest_hash,model_commit=cfg['model_commit']))
    sample_stop=threading.Event();gpu_samples=[]
    def monitor():
        while not sample_stop.is_set():
            try:
                rows=gpu_inventory();gpu_samples.append(rows)
                with (a.output/'gpu_samples.jsonl').open('a') as f:f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),gpus=rows))+'\n')
            except Exception as e:print('telemetry_error',str(e),flush=True)
            sample_stop.wait(2)
    if rank==0:threading.Thread(target=monitor,daemon=True).start()
    torch.cuda.reset_peak_memory_stats();model.train();dist.barrier();startup=time.monotonic()-launched;loop_started=time.monotonic()
    if rank==0:print(json.dumps(dict(startup_seconds=startup,trainable_parameters=trainable,frozen_parameters=frozen,mode=a.mode,schedule_input_tokens=cfg['target_input_tokens'])),flush=True)

    def checkpoint(current):
        begin=time.monotonic();folder=a.output/f'checkpoint-{current}';temp=a.output/f'checkpoint-{current}.partial'
        if rank==0:
            assert not folder.exists() and not temp.exists();temp.mkdir()
            tok.save_pretrained(temp)
        dist.barrier()
        with FSDP.state_dict_type(wrapped,StateDictType.LOCAL_STATE_DICT,LocalStateDictConfig(offload_to_cpu=False)):
            state=wrapped.state_dict()
            torch.save(dict(model=state,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=current,
                input_tokens=input_count,loss_tokens=loss_count,next_global_sequence=current*global_batch,
                timed_input=timed_input,timed_loss=timed_loss,timed_seconds=timed_seconds,parameter_digest=digest(wrapped),
                rng=dict(seen=seen,python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device))),temp/f'rank-{rank}.pt')
        del state
        dist.barrier()
        if rank==0:
            atomic_json(temp/'complete.json',dict(complete=True,step=current,mode=a.mode,config_sha256=config_hash,
                manifest_sha256=manifest_hash,runner_sha256=implementation_sha(Path(__file__)),files_sha256={p.name:sha(p) for p in sorted(temp.iterdir()) if p.is_file()}))
            temp.rename(folder)
        dist.barrier();torch.cuda.empty_cache();seconds=time.monotonic()-begin;checkpoint_times.append(seconds);return folder,seconds

    while step<stop:
        begin=time.monotonic();optimizer.zero_grad(set_to_none=True);local_input=0;local_loss=0;loss_sum=0.
        window_valid=sum(int((data[list(batch_indices(step,micro,rank,world,microbatch,global_batch)),1:]!=tok.pad_token_id).sum()) for micro in range(accumulation))
        window_targets=torch.tensor(window_valid,dtype=torch.float32,device=device);dist.all_reduce(window_targets)
        assert window_targets.item()>0
        for micro in range(accumulation):
            indices=batch_indices(step,micro,rank,world,microbatch,global_batch)
            assert len(seen)==step*(global_batch//world)+micro*microbatch;seen.extend(indices)
            batch=batch_from_tokens(data[indices.start:indices.stop],device,tok.pad_token_id)
            local_input+=(batch['input_ids']!=tok.pad_token_id).sum().item();valid=(batch['labels'][:,1:]!=-100).sum();local_loss+=valid.item()
            context=nullcontext() # synchronize each microbatch to keep full gradients sharded
            with context, torch.autocast('cuda',dtype=torch.bfloat16):
                loss=wrapped(**batch,num_items_in_batch=window_targets/world).loss;assert torch.isfinite(loss).item()
                loss_sum+=loss.detach().item();loss.backward()
        norm=wrapped.clip_grad_norm_(cfg['max_grad_norm']);assert torch.isfinite(norm).item()
        if a.mode=='preflight':
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params if p.numel())
            before_update=[p.detach().clone() for p in params]
        optimizer.step();step+=1
        if a.mode=='preflight' and step>1:
            assert all(not torch.equal(p,old) for p,old in zip(params,before_update) if p.numel())
        if a.mode=='preflight':del before_update
        if a.mode=='preflight':assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        monitor_start=time.monotonic()
        counts=torch.tensor([local_input,local_loss],dtype=torch.long,device=device);dist.all_reduce(counts)
        losses=torch.tensor(loss_sum,device=device);dist.all_reduce(losses);memory=all_memory(device)
        torch.cuda.synchronize();elapsed=torch.tensor(time.monotonic()-begin,device=device,dtype=torch.float64);dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
        seconds=elapsed.item();monitor_seconds=time.monotonic()-monitor_start
        input_count+=counts[0].item();loss_count+=counts[1].item();scheduler.step(input_count)
        if a.mode=='full' and input_count>=cfg['target_input_tokens']:stop=step
        if step>2:timed_input+=counts[0].item();timed_loss+=counts[1].item();timed_seconds+=seconds
        if rank==0:
            record=dict(step=step,input_tokens=input_count,loss_tokens=loss_count,step_input_tokens=counts[0].item(),step_loss_tokens=counts[1].item(),
                sequences=step*global_batch,step_seconds=seconds,monitor_seconds=monitor_seconds,loss=losses.item()/8,gradient_norm=norm.item(),lr=scheduler.get_last_lr()[0],
                timed_input=timed_input,timed_loss=timed_loss,timed_seconds=timed_seconds,
                input_tokens_per_second=timed_input/timed_seconds if timed_seconds else None,loss_tokens_per_second=timed_loss/timed_seconds if timed_seconds else None,
                gpu_memory=memory,elapsed_loop_seconds=time.monotonic()-loop_started)
            with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            atomic_json(a.output/'progress.json',record);print(json.dumps(record),flush=True)
        if step%cfg['save_steps']==0 or step==stop or (a.mode=='preflight' and step==2):
            last_checkpoint,save_seconds=checkpoint(step)
            if rank==0 and a.mode=='full':
                import shutil
                completed=sorted((p for p in a.output.glob('checkpoint-*') if (p/'complete.json').exists()),key=lambda p:int(p.name.split('-')[1]))
                for old in completed[:-2]:shutil.rmtree(old)
            dist.barrier()
    assert seen==expected_coverage(stop,rank,world,microbatch,global_batch)
    if a.mode=='full':assert input_count>=cfg['target_input_tokens']
    all_seen=[None]*8;dist.all_gather_object(all_seen,seen)
    assert len(set(x for part in all_seen for x in part))==sum(map(len,all_seen))==stop*global_batch
    reload_begin=time.monotonic()
    state=torch.load(last_checkpoint/f'rank-{rank}.pt',map_location='cpu',weights_only=False)
    assert digest(wrapped)==state['parameter_digest'];del state
    if a.compare_checkpoint:
        reference=torch.load(a.compare_checkpoint/f'rank-{rank}.pt',map_location='cpu',weights_only=False)
        current=torch.load(last_checkpoint/f'rank-{rank}.pt',map_location='cpu',weights_only=False)
        for key in ['step','input_tokens','loss_tokens','next_global_sequence','scheduler']:
            assert current[key]==reference[key]
        assert current['rng']['seen']==reference['rng']['seen']
        for key in ['torch','cuda']:assert torch.equal(current['rng'][key],reference['rng'][key])
        # FP32 master weights; tolerate only small BF16 reduction-order effects.
        maximum=0.
        assert current['model'].keys()==reference['model'].keys()
        for key,left in current['model'].items():
            right=reference['model'][key]
            if hasattr(left,'local_tensor'):left=left.local_tensor();right=right.local_tensor()
            assert torch.allclose(left,right,rtol=1e-5,atol=1e-7),key
            if left.numel():maximum=max(maximum,(left-right).abs().max().item())
        for pid,item in reference['optimizer']['state'].items():
            for key,value in item.items():
                other=current['optimizer']['state'][pid][key]
                if isinstance(value,torch.Tensor):assert torch.allclose(value,other,rtol=1e-3,atol=1e-7)
                else:assert value==other
        atomic_json(a.output/f'comparison_rank_{rank}.json',dict(passed=True,max_parameter_difference=maximum))
        del reference,current,left,right
    if a.mode=='full':
        with FSDP.state_dict_type(wrapped,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
            full_state=wrapped.state_dict()
            if rank==0:
                export=a.output/'final_model';model.save_pretrained(export,state_dict=full_state,max_shard_size='4GB');tok.save_pretrained(export)
                atomic_json(export/'checksums.json',{p.name:sha(p) for p in export.iterdir() if p.is_file()})
            del full_state
    dist.barrier();reload_seconds=time.monotonic()-reload_begin;memory=all_memory(device)
    if a.mode=='preflight':assert all(m['peak_allocated']<.9*m['total'] for m in memory), 'Insufficient live-tensor memory headroom'
    if rank==0:
        assert gpu_samples and all(any(sample[i][1]>0 for sample in gpu_samples) for i in range(8))
        result=dict(passed=True,mode=a.mode,optimizer_steps=step,input_tokens=input_count,loss_tokens=loss_count,timed_steps=max(0,step-2),
            timed_input=timed_input,timed_loss=timed_loss,timed_seconds=timed_seconds,
            input_tokens_per_second=timed_input/timed_seconds if timed_seconds else None,
            loss_tokens_per_second=timed_loss/timed_seconds if timed_seconds else None,
            startup_seconds=startup,training_loop_wall_seconds=time.monotonic()-loop_started,
            checkpoint_save_seconds=checkpoint_times,checkpoint_reload_seconds=reload_seconds,end_to_end_seconds=time.monotonic()-launched,
            final_loss=record['loss'],final_gradient_norm=record['gradient_norm'],checkpoint=str(last_checkpoint),
            checkpoint_reloaded=True,coverage_verified=True,finite_loss_and_gradients=True,all_text_causal_loss=True,
            all_parameters_checked_in_preflight=a.mode=='preflight',rank_assignments=assignments,gpu_memory=memory,
            config_sha256=config_hash,manifest_sha256=manifest_hash,runner_sha256=implementation_sha(Path(__file__)),
            trainable_parameters=trainable,frozen_parameters=frozen,monitoring='Async nvidia-smi every 2s; per-step scalar reductions and memory statistics included in timing; FP32 master parameters and AdamW states; BF16 FSDP compute, FP32 gradient reduction; local sharded checkpoints require eight ranks; retain two periodic checkpoints.')
        atomic_json(a.output/'result.json',result);print(json.dumps(result),flush=True)
    sample_stop.set();dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight-report',type=Path);p.add_argument('--resume-verification',type=Path)
    p.add_argument('--compare-checkpoint',type=Path)
    p.add_argument('--mode',choices=['preflight','pilot','full'],required=True);p.add_argument('--resume',type=Path);p.add_argument('--stop-after-step',type=int)
    main(p.parse_args())
