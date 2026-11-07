"""One-epoch Speedrun RSLoRA baseline, with mandatory separate preflight."""
from __future__ import annotations
import argparse
import fcntl
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import threading
from zoneinfo import ZoneInfo

import torch
import torch.distributed as dist
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed
from speedrun_format import training_row, make_collator, collator_check
from training_common import ensure_chat_template
from prepare_speedrun import digest, atomic_json, MODEL, REVISION
from validate_speedrun import file_sha


def gpu_inventory():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total',
                                     '--format=csv,noheader,nounits'],text=True)
    return [[int(x.strip()) for x in line.split(',')] for line in output.splitlines()]


def parameter_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode());h.update(p.detach().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def write_run_report(args,meta,validation,record,complete=False):
    run=args.output.resolve().parent
    review=run/'count_review.json'
    difference=(meta['eligible']-1190000)/1190000
    explanation=(json.loads(review.read_text())['explanation'] if review.exists() else
        ('The final count is within the 10% review threshold of the appendix estimate.' if abs(difference)<=.10 else
         'See the count-review evidence recorded for this dataset; the reference is an estimate, not a selection target.'))
    command_path=run/'baseline_command.txt'
    pipeline_status=run/'pipeline_status.json'
    phase=json.loads(pipeline_status.read_text()) if pipeline_status.exists() else {}
    gate=json.loads(args.preflight_report.read_text())
    result=dict(phase='complete' if complete else 'baseline_training',final_eligible=meta['eligible'],
        eligible_relative_difference=difference,eligible_count_explanation=explanation,
        master_path=str(args.dataset.resolve()/'master'),splits=validation['splits'],
        split0_input_token_budget=validation['splits'][0]['input_tokens'],
        split0_assistant_loss_token_budget=validation['splits'][0]['assistant_loss_tokens'],
        preflight_report=str(args.preflight_report.resolve()),preflight_passed=gate['passed'],
        preflight_checkpoint=gate['checkpoint_path'],validation_report=str(args.validation_report.resolve()),
        exact_training_command=command_path.read_text().strip() if command_path.exists() else None,
        tmux_session='speedrun_baseline',run_path=str(run),baseline_path=str(args.output.resolve()),
        log_path=phase.get('log'),checkpoint_path=record.get('checkpoint_path'),gpu_count=8,
        gpu_model=torch.cuda.get_device_name(),metrics=record,
        exact_one_epoch_coverage=complete,
        completed_utc=datetime.now(timezone.utc).isoformat() if complete else None)
    atomic_json(args.output.resolve()/'run_report.json',result)


def main(args):
    root=args.dataset.resolve();output=args.output.resolve()
    cfg=json.loads(args.config.read_text())
    config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest()
    approval=json.loads((root.parent/'training_approval.json').read_text())
    assert approval.get('approved') and approval.get('gpu_allocation_confirmed') and approval.get('reconstruction_settings_approved'), 'Training paused: recipe approval and project GPU allocation are required'
    meta=json.loads((root/'metadata.json').read_text())
    validation=json.loads(args.validation_report.read_text())
    assert validation['passed'] and validation['metadata_sha256']==file_sha(root/'metadata.json')
    assert validation['template_sha256']==meta['template_sha256']
    assert validation['format_sha256']==file_sha(Path(__file__).with_name('speedrun_format.py'))
    for relative,expected in validation['budget_files_sha256'].items():
        assert file_sha(args.validation_report.parent/relative)==expected
    assert meta['validated'] and meta['selected']==125000
    assert meta['model']==cfg['model'] and meta['model_revision']==cfg['revision']
    assert cfg['seed']==meta['seed'] and cfg['max_seq_length']==meta['max_length']
    assert cfg['epochs']==1
    assert cfg['per_device_train_batch_size']==1, 'This runner currently supports microbatch 1'
    assert cfg['global_batch_size']==32
    rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    assert world==len(approval['allocated_gpu_indices'])==8, 'Use the eight user-designated project GPUs'
    assert approval['allocated_gpu_indices']==list(range(8))
    assert 32%world==0
    if rank==0:
        training_lock=(root.parent/'speedrun_training.lock').open('a')
        fcntl.flock(training_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for relative,expected in meta['checksums'].items():
            assert file_sha(root/relative)==expected, f'Dataset checksum mismatch: {relative}'
        inventory=gpu_inventory()
        assert len(inventory)==world
        assert all(util<10 and used<2048 for _,util,used,total in inventory), f'GPUs are occupied: {inventory}'
        assert not output.exists(),f'Refusing to overwrite {output}'
        if not args.preflight:
            gate=json.loads(args.preflight_report.read_text())
            assert gate['passed'] and gate['world_size']==world
            assert gate['dataset_checksum']==meta['splits'][0]['id_sha256']
            assert gate['template_sha256']==meta['template_sha256']
            assert gate['training_config_sha256']==config_sha256
            assert gate['runner_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            assert gate['validation_report_sha256']==file_sha(args.validation_report)
            assert gate['checkpoint_reloaded'] and gate['all_gpus_active'] and gate['lora_synchronized']
            assert gate['cuda_allocator_config']==os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
        output.mkdir(parents=True)
    torch.cuda.set_device(local)
    dist.init_process_group('nccl',device_id=torch.device('cuda',local));dist.barrier()
    assert torch.cuda.is_bf16_supported()
    set_seed(cfg['seed'])
    tok=AutoTokenizer.from_pretrained(meta['tokenizer_model'],revision=meta['tokenizer_revision'])
    assert cfg['tokenizer_model']==meta['tokenizer_model'] and cfg['tokenizer_revision']==meta['tokenizer_revision']
    assert hashlib.sha256(tok.chat_template.encode()).hexdigest()==meta['template_sha256']
    raw=load_from_disk(root/'split_0')
    assert len(raw)==25000 and digest(raw['id'])==meta['splits'][0]['id_sha256']
    epoch_tokens=sum(raw['token_count']);assert epoch_tokens==meta['planned_split0_tokens']
    planned=epoch_tokens*cfg['epochs']
    expected_loss_tokens=validation['splits'][0]['assistant_loss_tokens']
    # The only permutation, independently identical on all ranks; modulo assignment never pads.
    order=torch.randperm(len(raw),generator=torch.Generator().manual_seed(cfg['seed'])).tolist()
    assigned=order[rank::world]
    rank_assignments=[None]*world
    dist.all_gather_object(rank_assignments,dict(rank=rank,local_rank=local,device=torch.cuda.current_device(),
        gpu_uuid=str(torch.cuda.get_device_properties(local).uuid)))
    assert {x['rank'] for x in rank_assignments}==set(range(8))
    assert {x['local_rank'] for x in rank_assignments}==set(range(8))
    assert len({x['gpu_uuid'] for x in rank_assignments})==8
    all_ids=[None]*world;dist.all_gather_object(all_ids,[raw[i]['id'] for i in assigned])
    flattened=[x for part in all_ids for x in part]
    assert len(flattened)==len(set(flattened))==25000
    assert len({len(part) for part in all_ids})==1
    if args.preflight:
        # Probe the longest local examples, bounding the full epoch's activation footprint.
        assigned=sorted(assigned,key=lambda i:(-raw[i]['token_count'],i))[:12]
    else:
        assigned=assigned*cfg['epochs']
    collator=make_collator(tok)
    collator_check(tok,[training_row(tok,raw[i]) for i in assigned[:4]])
    model=AutoModelForCausalLM.from_pretrained(cfg['model'],revision=meta['model_commit'],dtype=torch.bfloat16)
    model.requires_grad_(False);model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=cfg['r'],lora_alpha=cfg['lora_alpha'],
        use_rslora=cfg['use_rslora'],lora_dropout=cfg['lora_dropout'],target_modules=cfg['target_modules'],
        bias='none',task_type='CAUSAL_LM'))
    model.enable_input_require_grads()
    if cfg['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
    model.to(local);wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local],find_unused_parameters=False)
    params=[p for p in model.parameters() if p.requires_grad]
    assert cfg['optim']=='adamw_torch', 'Only AdamW is implemented in this baseline runner'
    optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],
        betas=(cfg['adam_beta1'],cfg['adam_beta2']),eps=cfg['adam_epsilon'])
    accum=cfg['global_batch_size']//world;total_steps=math.ceil(len(assigned)/accum)
    scheduler=get_scheduler(cfg['lr_scheduler_type'],optimizer=optimizer,
        num_warmup_steps=math.ceil(total_steps*cfg['warmup_ratio']),num_training_steps=total_steps)
    atomic_json(output/f'config_rank_{rank}.json',dict(settings=cfg,training_config_sha256=config_sha256,
        world_size=world,max_train_tokens=None,max_steps=None,bf16=True,completion_only_loss=True,
        planned_tokens=planned,local_examples=len(assigned),id_checksum=digest([raw[i]['id'] for i in assigned])))
    start=time.monotonic();processed=0;loss_tokens=0;examples=0;timed_tokens=0;timed_loss_tokens=0;timed_seconds=0;step=0
    local_seen=[];active=False;checkpoint=None
    torch.cuda.reset_peak_memory_stats();wrapped.train()
    sampling_stop=threading.Event();samples=[];sampling_errors=[]
    def sample_gpus():
        while not sampling_stop.is_set():
            try:
                inventory=gpu_inventory();samples.append((time.monotonic(),inventory))
                with (output/'gpu_samples.jsonl').open('a') as f:
                    f.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),gpus=inventory))+'\n')
            except Exception as exc:sampling_errors.append(str(exc))
            sampling_stop.wait(2)
    if rank==0:threading.Thread(target=sample_gpus,daemon=True).start()
    if rank==0:
        print(json.dumps(dict(preliminary_seconds=planned/cfg['throughput_reference']['input_tokens_per_second'],throughput_reference=cfg['throughput_reference'],planned_input_tokens=planned)),flush=True)
    for offset in range(0,len(assigned),accum):
        window=assigned[offset:offset+accum];optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();tick=time.monotonic();counts=torch.zeros(3,dtype=torch.long,device=local);loss_sum=0.
        for micro,index in enumerate(window):
            row=training_row(tok,raw[index]);local_seen.append(raw[index]['id'])
            batch={k:v.to(local) for k,v in collator([row]).items()}
            counts[0]+=batch['attention_mask'].sum();counts[1]+=(batch['labels']!=-100).sum()
            assert (batch['labels']!=-100).sum().item()>0
            counts[2]+=1
            sync=wrapped.no_sync() if micro+1<len(window) else nullcontext()
            with sync:
                loss=wrapped(**batch).loss
                assert torch.isfinite(loss).item(),'Nonfinite loss'
                loss_sum+=loss.detach().item()
                (loss/len(window)).backward()
        norm=torch.nn.utils.clip_grad_norm_(params,cfg['max_grad_norm'])
        assert torch.isfinite(norm).item(),'Nonfinite gradient norm'
        optimizer.step();scheduler.step();step+=1
        torch.cuda.synchronize();elapsed=time.monotonic()-tick
        dist.all_reduce(counts,op=dist.ReduceOp.SUM)
        values=torch.tensor([loss_sum/len(window),elapsed],device=local,dtype=torch.float64)
        dist.all_reduce(values[:1],op=dist.ReduceOp.SUM);dist.all_reduce(values[1:],op=dist.ReduceOp.MAX)
        processed+=counts[0].item();loss_tokens+=counts[1].item();examples+=counts[2].item()
        if step>2:
            timed_tokens+=counts[0].item();timed_loss_tokens+=counts[1].item();timed_seconds+=values[1].item()
        if args.preflight or step<=3:
            hashes=[None]*world;dist.all_gather_object(hashes,parameter_digest(model))
            assert len(set(hashes))==1,'LoRA parameters differ across ranks'
        memory=[None]*world
        dist.all_gather_object(memory,dict(rank=rank,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            total=torch.cuda.get_device_properties(local).total_memory))
        if rank==0:
            inventory=gpu_inventory()
            recent=[inv for stamp,inv in samples if stamp>=tick]+[inventory]
            utilization={i:max(inv[i][1] for inv in recent) for i in range(world)}
            active=active or all(utilization[i]>0 for i in range(world))
            rate=timed_tokens/timed_seconds if timed_seconds else None
            loss_rate=timed_loss_tokens/timed_seconds if timed_seconds else None
            now=datetime.now(ZoneInfo('America/New_York'))
            midnight=datetime.combine(now.date()+timedelta(days=1),datetime.min.time(),tzinfo=now.tzinfo)
            remaining=(planned-processed)/rate if rate else None
            record=dict(optimizer_step=step,examples_processed=examples,global_non_padding_input_tokens=processed,completion_loss_tokens=loss_tokens,
                step_input_tokens=counts[0].item(),step_assistant_loss_tokens=counts[1].item(),step_examples=counts[2].item(),
                step_seconds=values[1].item(),timed_input_tokens=timed_tokens,timed_assistant_loss_tokens=timed_loss_tokens,timed_seconds=timed_seconds,
                elapsed_training_seconds=time.monotonic()-start,weighted_tokens_per_second=rate,loss=values[0].item()/world,
                weighted_assistant_loss_tokens_per_second=loss_rate,gpu_utilization_peak_during_step=utilization,
                learning_rate=scheduler.get_last_lr()[0],gradient_norm=norm.item(),gpu_memory=memory,gpu_inventory=inventory,
                remaining_seconds=remaining,estimated_completion_time=(now+timedelta(seconds=remaining)).isoformat() if remaining else None,
                estimated_tokens_by_end_of_day=min(planned,int(processed+rate*(midnight-now).total_seconds())) if rate else None,
                estimated_assistant_loss_tokens_by_end_of_day=min(expected_loss_tokens,int(loss_tokens+loss_rate*(midnight-now).total_seconds())) if loss_rate else None,
                checkpoint_path=str(checkpoint) if checkpoint else None)
            with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            atomic_json(output/'progress.json',record);print(json.dumps(record),flush=True)
        saving=(args.preflight and step==3) or step%cfg['save_steps']==0 or offset+len(window)==len(assigned)
        if saving:
            checkpoint=output/f'checkpoint-{step}'
            if rank==0:
                model.save_pretrained(checkpoint);tok.save_pretrained(checkpoint)
                torch.save(dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,
                    input_tokens=processed,assistant_loss_tokens=loss_tokens,examples_processed=examples,
                    dataset_position_per_rank=offset+len(window),training_config_sha256=config_sha256,
                    dataset_checksum=meta['splits'][0]['id_sha256'],validation_report_sha256=file_sha(args.validation_report),
                    timed_tokens=timed_tokens,timed_loss_tokens=timed_loss_tokens,timed_seconds=timed_seconds),checkpoint/'training_state.pt')
                from safetensors.torch import load_file
                adapter=load_file(str(checkpoint/'adapter_model.safetensors'))
                assert adapter and all(torch.isfinite(x).all().item() for x in adapter.values())
                record['checkpoint_path']=str(checkpoint)
                atomic_json(output/'progress.json',record)
                print(json.dumps(dict(checkpoint_path=str(checkpoint),optimizer_step=step)),flush=True)
            dist.barrier()
            import random
            import numpy as np
            torch.save(dict(rank=rank,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),seen_ids=local_seen),checkpoint/f'rank_{rank}_state.pt')
            dist.barrier()
            if rank==0:
                atomic_json(checkpoint/'checkpoint_manifest.json',dict(complete=True,optimizer_step=step,
                    examples_processed=examples,input_tokens=processed,assistant_loss_tokens=loss_tokens,
                    dataset_position_per_rank=offset+len(window),dataset_checksum=meta['splits'][0]['id_sha256'],
                    training_config_sha256=config_sha256,world_size=world,
                    files_sha256={p.name:file_sha(p) for p in sorted(checkpoint.iterdir()) if p.is_file()}))
            dist.barrier()
        if args.preflight and step==3:
            from safetensors.torch import load_file
            before=parameter_digest(model)
            # Alter parameters first, then restore the saved adapter through PEFT's loader.
            with torch.no_grad():
                for p in params:p.zero_()
            restored=set_peft_model_state_dict(model,load_file(str(checkpoint/'adapter_model.safetensors')))
            assert not restored.unexpected_keys
            assert parameter_digest(model)==before, 'Adapter checkpoint reload mismatch'
            if rank==0:
                state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
                assert state['step']==3 and state['input_tokens']==processed and state['assistant_loss_tokens']==loss_tokens
                assert state['optimizer']['state'] and state['scheduler']
                assert all(torch.isfinite(v).all().item() for item in state['optimizer']['state'].values()
                           for v in item.values() if isinstance(v,torch.Tensor))
                del state
            model.eval()
            with torch.no_grad():
                probe={k:v.to(local) for k,v in collator([training_row(tok,raw[assigned[0]])]).items()}
                reload_loss=model(**probe).loss
                assert torch.isfinite(reload_loss).item(), 'Nonfinite loss after checkpoint reload'
            del probe
            reload_results=[None]*world
            dist.all_gather_object(reload_results,dict(rank=rank,checkpoint_reloaded=True,loss=reload_loss.item()))
            dist.all_gather_object(memory,dict(rank=rank,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                total=torch.cuda.get_device_properties(local).total_memory))
            if rank==0:
                safe=all(x['peak_reserved']<.90*x['total'] for x in memory)
                # Include loading, tokenizer/collator, synchronization and checkpoint gates.
                assert active,'Not every GPU showed activity'
                assert safe,'Insufficient projected memory headroom'
                atomic_json(output/'preflight.json',dict(passed=True,world_size=world,
                    dataset_checksum=meta['splits'][0]['id_sha256'],template_sha256=meta['template_sha256'],training_config_sha256=config_sha256,
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    validation_report_sha256=file_sha(args.validation_report),checkpoint_reloaded=True,
                    cuda_allocator_config=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
                    checkpoint_reload_results=reload_results,rank_assignments=rank_assignments,
                    gpu_utilization_peak_during_step=utilization,gpu_sampling_errors=sampling_errors,
                    unique_examples_across_ranks=True,finite_loss_and_gradients=True,lora_synchronized=True,
                    all_gpus_active=active,checkpoint_path=str(checkpoint),gpu_memory=memory,
                    probe='Three optimizer steps on the twelve longest assigned sequences per rank.'))
            break
        if not args.preflight and (step in [5,10] or saving):
            if rank==0:write_run_report(args,meta,validation,record)
    if not args.preflight:
        assert processed==planned, f'Epoch token count differs: {processed} != {planned}'
        assert loss_tokens==expected_loss_tokens
        assert examples==25000 and step==782
        assert local_seen==[raw[i]['id'] for i in assigned]
        all_seen=[None]*world;dist.all_gather_object(all_seen,local_seen)
        assert len([id_ for part in all_seen for id_ in part])==len(set(id_ for part in all_seen for id_ in part))==25000
        if rank==0:
            atomic_json(output/'completion.json',dict(complete=True,exact_one_epoch_coverage=True,
                optimizer_steps=step,examples_processed=examples,input_tokens=processed,assistant_loss_tokens=loss_tokens,
                elapsed_training_seconds=time.monotonic()-start,weighted_input_tokens_per_second=timed_tokens/timed_seconds,
                weighted_assistant_loss_tokens_per_second=timed_loss_tokens/timed_seconds,final_loss=record['loss'],
                checkpoint_path=str(checkpoint),gpu_memory=memory))
            write_run_report(args,meta,validation,record,complete=True)
    sampling_stop.set()
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[2]/'em_influence_examples/training/speedrun_baseline.json')
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--preflight-report',type=Path)
    parser.add_argument('--validation-report',type=Path,required=True)
    main(parser.parse_args())
