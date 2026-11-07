"""Summarize measured continued-midtraining throughput and prepare, but never launch, full training."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import statistics
import shlex
import sys
import subprocess
from midtrain_common import BASE,atomic_json,sha,implementation_sha


def main(run):
    cfg=json.loads((run/'config.json').read_text());result=json.loads((run/'pilot/result.json').read_text())
    resume=json.loads((run/'resume_verification.json').read_text());assert result['passed'] and resume['passed']
    cache=Path(cfg.get('dataset_dir',BASE/'pilot'))
    manifest=json.loads((cache/'manifest.json').read_text())
    metrics=[json.loads(l) for l in (run/'pilot/metrics.jsonl').read_text().splitlines()]
    samples=[json.loads(l) for l in (run/'pilot/gpu_samples.jsonl').read_text().splitlines()]
    timed=[m for m in metrics if m['step']>2]
    rate=sum(m['step_input_tokens'] for m in timed)/sum(m['step_seconds'] for m in timed)
    assert math.isclose(rate,result['input_tokens_per_second'],rel_tol=1e-12)
    target=cfg['target_input_tokens'];input_per_step=result['input_tokens']/result['optimizer_steps']
    expected_steps=math.ceil(target/input_per_step);save_count=math.ceil(expected_steps/cfg['save_steps'])
    save_seconds=statistics.mean(result['checkpoint_save_seconds'])
    compute_seconds=target/rate;startup=result['startup_seconds'];checkpoint_seconds=save_seconds*save_count
    process_observation=run/'pilot_process_observation.json'
    if process_observation.exists():
        observation=json.loads(process_observation.read_text())
        first_sample=datetime.fromisoformat(samples[0]['utc']).timestamp()
        startup=max(startup,first_sample-observation['process_created_epoch'])
        done=json.loads((run/'status.json').read_text())
        if done['phase']=='pilot_completed':
            result['process_end_to_end_seconds']=datetime.fromisoformat(done['utc']).timestamp()-observation['process_created_epoch']
        result['startup_including_process_imports_seconds']=startup
        result['startup_measurement_note']='Torchrun process creation to first async GPU sample after model/DDP readiness; approximately 0.2s sampling/query uncertainty.'
    total_seconds=compute_seconds+startup+checkpoint_seconds
    def capacity(seconds):
        available=max(0,seconds-startup)
        lo=0;hi=int(available*rate)
        while lo<hi:
            mid=(lo+hi+1)//2;steps=math.ceil(mid/input_per_step);saves=math.ceil(steps/cfg['save_steps'])
            if mid/rate+saves*save_seconds<=available:lo=mid
            else:hi=mid-1
        return lo
    periods=[m['step_seconds'] for m in timed]
    import numpy as np
    gpu=[]
    times=[datetime.fromisoformat(s['utc']).timestamp() for s in samples]
    span=times[-1]-times[0]
    for i,memory in enumerate(result['gpu_memory']):
        utilization=[next(row[1] for row in s['gpus'] if row[0]==i) for s in samples]
        mean=sum((utilization[j]+utilization[j+1])/2*(times[j+1]-times[j]) for j in range(len(times)-1))/span
        gpu.append(dict(**memory,mean_utilization_percent=mean,max_utilization_percent=max(utilization)))
    full_command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',
        str(Path(__file__).with_name('train_midtrain.py')),'--config',str(run/'config.json'),'--mode','full',
        '--output','<NEW_TIMESTAMPED_FULL_RUN_DIRECTORY>','--preflight-report',str(run/'preflight/result.json'),
        '--resume-verification',str(run/'resume_verification.json')]
    environment=json.loads((run/'environment.json').read_text())
    cmd=shlex.join(['env',*[f'{k}={v}' for k,v in environment.items()],*full_command])
    (run/'full_command.txt').write_text(cmd+'\n')
    script=run/'launch_full_after_approval.py'
    launcher = """#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,shlex
run=Path(RUN_VALUE)
base=Path(BASE_VALUE)
config=run/'config.json'
approval=json.loads((base/'full_run_approval.json').read_text())
assert approval.get('approved') and approval['config_sha256']==hashlib.sha256(config.read_bytes()).hexdigest(),'Separate full-run approval required'
checks=json.loads((run/'checksums.json').read_text())
for name,expected in checks['source_files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,'Scientific source changed; repeat validation before launch'
subprocess.run(PREPARE_COMMAND,check=True)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
output=base/'runs'/('full_'+stamp)
command=COMMAND_VALUE
command[command.index('--output')+1]=str(output)
environment=ENVIRONMENT_VALUE
log=output.with_suffix('.log')
shell='exec '+shlex.join(['env',*[f'{k}={v}' for k,v in environment.items()],*command])+' > '+shlex.quote(str(log))+' 2>&1'
output.with_suffix('.launch.json').write_text(json.dumps(dict(command=command,environment=environment,log=str(log)),indent=2)+'\\n')
subprocess.run(['tmux','new-session','-d','-s','midtrain_full_'+stamp,shell],check=True)
print(json.dumps(dict(output=str(output),log=str(log),session='midtrain_full_'+stamp),indent=2))
"""
    launcher=launcher.replace('RUN_VALUE',repr(str(run))).replace('BASE_VALUE',repr(str(BASE)))
    launcher=launcher.replace('PREPARE_COMMAND',repr([sys.executable,str(Path(__file__).with_name('prepare_midtrain.py')),'--materialize','full']))
    launcher=launcher.replace('COMMAND_VALUE',repr(full_command)).replace('ENVIRONMENT_VALUE',repr(environment))
    compile(launcher,str(script),'exec')
    script.write_text(launcher)
    # Keep an exact concrete command template and an explicit resume command as separate artifacts.
    resume_command=full_command+['--resume','<COMPLETE_FULL_RUN_CHECKPOINT>']
    (run/'full_resume_command.txt').write_text(shlex.join(['env',*[f'{k}={v}' for k,v in environment.items()],*resume_command])+'\n')
    approval=BASE/'full_run_approval.json'
    if not approval.exists():atomic_json(approval,dict(approved=False,config_sha256=sha(run/'config.json'),reason='User requested pilot only; full run not authorized.'))
    guard_check=subprocess.run([sys.executable,str(script)],capture_output=True,text=True)
    assert guard_check.returncode!=0 and 'Separate full-run approval required' in guard_check.stderr
    estimate=dict(training_loop_seconds=compute_seconds,startup_seconds=startup,projected_checkpoint_saves=save_count,
        measured_seconds_per_checkpoint=save_seconds,projected_checkpoint_seconds=checkpoint_seconds,
        total_seconds=total_seconds,hours=total_seconds/3600,days=total_seconds/86400,gpu_hours=total_seconds*8/3600,
        planning_contingency_fraction=.20,planning_hours_20pct=total_seconds*1.2/3600,planning_days_20pct=total_seconds*1.2/86400,
        planning_gpu_hours_20pct=total_seconds*1.2*8/3600,expected_optimizer_steps=expected_steps,
        expected_last_step_rounding_upper_bound_input_tokens=cfg['sequence_length']*cfg['global_batch_sequences'],
        input_tokens_24h=capacity(86400),input_tokens_7d=capacity(7*86400),
        capacity_assumption='Uncapped rate projection for sustained work, including estimated startup and checkpoint overhead; full run itself stops at 1B non-padding input tokens.',
        excludes='Full data-cache construction/network time, which must finish and validate before the scientific training timer starts.')
    final=dict(verified_stage='Olmo 3 7B stage 2 midtraining',official_stage_token_budget=100_000_000_000,
        published_budget_note='Official stage budget counts token positions; selected cached arrays contain existing padding. Project target is 1B non-padding input tokens, not 1% of documents.',
        dataset=cfg['dataset'],dataset_revision=cfg['dataset_commit'],official_manifest=str(BASE/'research/OLMo-midtraining-mix-0625-100B.txt'),
        source_weights=str(BASE/'source_weights.json'),packing_policy=manifest['packing_policy'],packing_attention_clarification=str(BASE/'packing_policy_clarification.json'),
        initialization=dict(model=cfg['model'],revision=cfg['model_revision_name'],commit=cfg['model_commit'],approved=True),
        config=str(run/'config.json'),pilot=result,estimate=estimate,gpu=gpu,
        step_seconds=dict(minimum=min(periods),median=statistics.median(periods),p95=float(np.percentile(periods,95)),maximum=max(periods),stdev=statistics.stdev(periods)),
        mean_step_monitor_seconds=statistics.mean(m['monitor_seconds'] for m in timed),
        monitoring_overhead_note='Post-update synchronization/statistics section is included in timed throughput and includes trailing optimizer GPU work; it is an upper bound, not an isolated A/B telemetry overhead measurement. Async nvidia-smi samples every 2s.',
        resume_verification=resume,pilot_manifest=str(cache/'manifest.json'),pilot_cache=str(cache/'tokens.u32'),
        full_plan=str(BASE/'full_plan.json'),full_run_command_file=str(run/'full_command.txt'),
        full_run_launcher=str(script),full_resume_command_file=str(run/'full_resume_command.txt'),
        full_run_launched=False,full_launch_approval_guard_tested=True,remaining_blocker='Separate user approval is required before preparing/launching the full scientific run.',
        uncertainties=['Pilot is fully cached; full data acquisition time is not measured. HTTP object ETags and cached byte hashes pin external token arrays.',
            'Per-source official window-slot weights are preserved; observed non-padding source-token shares differ where official PDF streams contain PADs.',
            'Short pilot, variable padding, future allocator growth and checkpoint-state growth introduce uncertainty.',
            'Scheduler uses actual cumulative non-padding tokens with 3% warmup and linear decay; pilot stops early on this full-budget schedule. Full run initializes fresh.'])
    atomic_json(run/'report.json',final)
    hashes=dict(source_files={str(Path(__file__).with_name(n)):sha(Path(__file__).with_name(n)) for n in ['train_midtrain.py','midtrain_common.py','midtrain_optimizations.py','prepare_midtrain.py']},config_sha256=sha(run/'config.json'),runner_sha256=implementation_sha(Path(__file__).with_name('train_midtrain.py')),
        official_manifest_sha256=sha(BASE/'research/OLMo-midtraining-mix-0625-100B.txt'),inventory_sha256=sha(BASE/'inventory.jsonl'),
        source_weights_sha256=sha(BASE/'source_weights.json'),pilot_manifest_sha256=sha(cache/'manifest.json'),pilot_tokens_sha256=sha(cache/'tokens.u32'),
        full_plan_sha256=sha(BASE/'full_plan.json'),pilot_checkpoint=json.loads((Path(result['checkpoint'])/'complete.json').read_text()))
    atomic_json(run/'checksums.json',hashes)
    (run/'report.md').write_text(f'''# LoRA continued-midtraining throughput pilot

Approved initialization: `{cfg['model_revision_name']}` (`{cfg['model_commit']}`), before mid-training. No Speedrun adapter was loaded.

Official stage: 100B published token positions; project target: 1B non-padding input tokens. Dataset: `{cfg['dataset']}` at `{cfg['dataset_commit']}`. Use the pinned official 7B 0625 pretokenized mixture, not an older Dolmino release. Exact source-window weights and per-source observed non-padding token counts are saved in the sample artifacts. Native model attention crosses document/PAD boundaries without extra document resets (architecture sliding-window restrictions remain unchanged); EOS is preserved; PAD and first-position targets are masked. Long documents are chunked, not discarded.

Hardware: eight NVIDIA A100-SXM4-40GB GPUs. BF16 autocast, FlashAttention 2, fused linear cross-entropy, nonreentrant checkpointing, frozen base, RSLoRA rank/alpha 64, seven projection types, microbatch {cfg['microbatch']}, accumulation {cfg['gradient_accumulation']}, global {cfg['global_batch_sequences']}, context {cfg['sequence_length']}. Fused AdamW LR peak {cfg['learning_rate']:g}, betas {cfg['betas']}, epsilon 1e-8, weight decay .01, clipping 1, seed 0. Full-budget token-based linear schedule with 3% warmup, stopped early for the pilot.

- Trainable / frozen parameters: {result['trainable_parameters']:,} / {result['frozen_parameters']:,}.
- Actual pilot input / loss tokens: {result['input_tokens']:,} / {result['loss_tokens']:,}.
- Timed steps: {result['timed_steps']}; first two excluded.
- Timed seconds: {result['timed_seconds']:.3f}.
- Weighted input throughput: **{rate:,.2f} tokens/s**.
- Weighted loss throughput: {result['loss_tokens_per_second']:,.2f} tokens/s.
- Step time min / median / p95 / max: {min(periods):.3f} / {statistics.median(periods):.3f} / {float(np.percentile(periods,95)):.3f} / {max(periods):.3f} seconds.
- Mean post-update synchronization/statistics section: {statistics.mean(m['monitor_seconds'] for m in timed)*1000:.2f} ms/step, included in throughput. This includes trailing optimizer GPU work; telemetry overhead was not isolated in an A/B run. Async GPU sampling every 2s; no timed-loop parameter hashes.
- Startup: {startup:.3f}s; checkpoint saves: {result['checkpoint_save_seconds']}; reload: {result['checkpoint_reload_seconds']:.3f}s; end-to-end process wall: {result.get('process_end_to_end_seconds',result['end_to_end_seconds']):.3f}s (main-function wall: {result['end_to_end_seconds']:.3f}s).
- Final loss / gradient norm: {result['final_loss']:.6f} / {result['final_gradient_norm']:.6f}.
- Eight-GPU preflight, adapter reload, and interrupted-versus-uninterrupted resume comparison: passed. Resume maximum parameter difference: {resume['uninterrupted_vs_resumed_max_parameter_difference']:.9g}.

| GPU | Mean utilization | Peak allocated GiB | Peak reserved GiB |
| --- | ---: | ---: | ---: |
'''+''.join(f"| {g['rank']} | {g['mean_utilization_percent']:.1f}% | {g['peak_allocated']/2**30:.2f} | {g['peak_reserved']/2**30:.2f} |\n" for g in gpu)+f'''
Full target estimate: **{estimate['hours']:.2f} hours / {estimate['days']:.2f} days**, {estimate['gpu_hours']:.1f} GPU-hours, approximately {expected_steps:,} optimizer steps. This adds measured startup and {save_count} projected checkpoint saves to target tokens / measured input throughput. A separately labeled **20% contingency** gives **{estimate['planning_hours_20pct']:.2f} hours**, {estimate['planning_gpu_hours_20pct']:.1f} GPU-hours.

Uncapped projected capacity: {estimate['input_tokens_24h']:,} input tokens in 24h; {estimate['input_tokens_7d']:,} in seven days. The actual scientific run will stop at the first completed optimizer step reaching 1B non-padding input tokens. Full-cache download time is excluded and remains uncertain. A 6% overprovisioned source-stratified window plan is prepared; the full cache has not been materialized.

Pilot checkpoint: `{result['checkpoint']}`. Full command: `full_command.txt`; approval-guarded launcher: `launch_full_after_approval.py`; resume command: `full_resume_command.txt`. Checkpoint completion markers are atomic; resume restores adapter, optimizer/scheduler, rank RNG, exact sampler position and token counters. Do not initialize the scientific run from the pilot.

**Stopped after the pilot. The full scientific run has not been launched and requires user approval.** This is a throughput feasibility measurement, not an attribution or filtering evaluation.
''')
    print(json.dumps(dict(input_tokens=result['input_tokens'],loss_tokens=result['loss_tokens'],input_rate=rate,estimate=estimate),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run.resolve())
