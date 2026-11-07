"""Gate and launch Speedrun mid-training full-parameter, preserving the LoRA artifacts."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from midtrain_common import BASE, atomic_json, sha, batch_layout, training_prefix


def main(a):
    run=a.run.resolve();run.mkdir(parents=True,exist_ok=False)
    scripts=Path(__file__).resolve().parent
    config=run/'config.json';config.write_bytes(a.config.read_bytes())
    cfg=json.loads(config.read_text());global_batch,_,_=batch_layout(cfg,8)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    atomic_json(run/'environment.json', {k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','TOKENIZERS_PARALLELISM','PYTORCH_CUDA_ALLOC_CONF']})
    for name,cmd in [('gpu_inventory',['nvidia-smi','-q']),('topology',['nvidia-smi','topo','-m']),('packages',[sys.executable,'-m','pip','freeze'])]:
        (run/f'{name}.txt').write_text(subprocess.check_output(cmd,text=True))
    shutil.copy2(scripts/'train_midtrain_full.py',run/'runner_source.py')
    (run/'source').mkdir()
    for name in ['train_midtrain_full.py','midtrain_common.py','midtrain_optimizations.py']:
        shutil.copy2(scripts/name,run/'source'/name)
    cache=Path(cfg.get('dataset_dir',BASE/'full'));manifest=cache/'manifest.json'
    shutil.copy2(manifest,run/'dataset_manifest.json')
    data=json.loads(manifest.read_text());assert sha(cache/'tokens.u32')==data['tokens_sha256']
    prefix,total=training_prefix(data['entries'],cfg['target_input_tokens'],global_batch)
    atomic_json(run/'data_audit.json',dict(manifest_sha256=sha(manifest),tokens_sha256=data['tokens_sha256'],ordered_entries_sha256=__import__('midtrain_common').object_sha(data['entries']),
        input_tokens=total,loss_tokens=sum(x['loss_tokens'] for x in prefix),steps=len(prefix)//global_batch,recipe=cfg,
        baseline=str(BASE/'runs/full_20261026T042748Z'),retention='Keep all validation artifacts; keep latest two complete full-run checkpoints, plus final Hugging Face export. Only this new run is pruned.'))
    def status(phase,**kw):
        atomic_json(run/'status.json',dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),**kw));print(phase,kw,flush=True)
    def execute(name,mode,extra=()):
        cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(scripts/'train_midtrain_full.py'),'--config',str(config),'--mode',mode,'--output',str(run/name),*extra]
        (run/f'{name}_command.txt').write_text(shlex.join(cmd)+'\n');status(name)
        started=time.monotonic()
        with (run/f'{name}.log').open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        result=json.loads((run/name/'result.json').read_text());result['process_wall_seconds']=time.monotonic()-started
        atomic_json(run/name/'result.json',result)
        assert result['passed'] and result['checkpoint_reloaded'] and result['coverage_verified']
        return result
    try:
        gate=execute('preflight','preflight')
        resume=execute('resume_probe','preflight',['--resume',str(run/'preflight/checkpoint-2'),'--stop-after-step','3'])
        for rank in range(8):
            restored=json.loads((run/f'resume_probe/resume_rank_{rank}.json').read_text())
            assert all(restored[k] for k in ['coverage_prefix_verified','rng_verified','model_exactly_restored','optimizer_exactly_restored','scheduler_restored'])

        execute('replay_control','preflight')
        status('resume_calibration')
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(scripts/'audit_midtrain_full_resume.py'),'--pipeline-dir',str(run)]
        with (run/'resume_calibration.log').open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        assert json.loads((run/'resume_verification.json').read_text())['passed']
        args=['--preflight-report',str(run/'preflight/result.json'),'--resume-verification',str(run/'resume_verification.json')]
        pilot=execute('pilot','pilot',args);assert pilot['timed_steps']>=30 and pilot['input_tokens']>=10000000
        rows=[json.loads(line) for line in (run/'pilot/metrics.jsonl').read_text().splitlines()][2:]
        import statistics
        loop=total/pilot['input_tokens_per_second'];save=statistics.mean(pilot['checkpoint_save_seconds'])
        estimate=loop+pilot['startup_seconds']+save*((len(prefix)//global_batch+cfg['save_steps']-1)//cfg['save_steps'])
        atomic_json(run/'pilot_estimate.json',dict(loop_hours=loop/3600,loop_days=loop/86400,loop_gpu_hours=8*loop/3600,
            estimated_total_hours=estimate/3600,estimated_total_days=estimate/86400,estimated_total_gpu_hours=8*estimate/3600,
            checkpoint_seconds=save,startup_seconds=pilot['startup_seconds'],step_seconds_min=min(x['step_seconds'] for x in rows),step_seconds_max=max(x['step_seconds'] for x in rows),step_seconds_stdev=statistics.stdev(x['step_seconds'] for x in rows),
            pilot_overshoot_tokens=pilot['input_tokens']-10000000))
        size=lambda folder:sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
        checkpoint_size=size(Path(pilot['checkpoint']));needed=checkpoint_size*3+40_000_000_000
        free=shutil.disk_usage(run).free;assert free>needed,(free,needed)
        atomic_json(run/'disk_plan.json',dict(checkpoint_bytes=checkpoint_size,free_bytes=free,required_bytes=needed,retained_periodic_checkpoints=2,temporary_checkpoint_slots=1,final_export_allowance=40_000_000_000))
        result=execute('full','full',args)
        status('independent_final_reload')
        subprocess.run([sys.executable,str(scripts/'verify_midtrain_full.py'),'--run',str(run)],env=env,check=True)
        status('complete',result=result)
    except Exception as exc:status('failed',error=str(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--config',type=Path,required=True);main(p.parse_args())
