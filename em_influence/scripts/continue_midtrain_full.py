"""Continue a stopped pipeline using independent replay calibration, then pilot and fresh full run."""
from datetime import datetime,timezone
import json
from pathlib import Path
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from midtrain_common import atomic_json,sha,implementation_sha
run=Path(sys.argv[1]).resolve();scripts=Path(__file__).resolve().parent
cfg=json.loads((run/'config.json').read_text())
env=os.environ.copy();env.update(json.loads((run/'environment.json').read_text()))
def status(phase,**kw):
    atomic_json(run/'status.json',dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),**kw));print(phase,kw,flush=True)
def execute(name,mode,extra=()):
    cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(scripts/'train_midtrain_full.py'),'--config',str(run/'config.json'),'--mode',mode,'--output',str(run/name),*extra]
    (run/f'{name}_command.txt').write_text(shlex.join(cmd)+'\n');status(name);start=time.monotonic()
    with (run/f'{name}.log').open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    result=json.loads((run/name/'result.json').read_text());result['process_wall_seconds']=time.monotonic()-start
    atomic_json(run/name/'result.json',result)
    assert result['passed'] and result['coverage_verified'];return result
try:
    gate=json.loads((run/'preflight/result.json').read_text());assert gate['passed'] and gate['runner_sha256']==implementation_sha(scripts/'train_midtrain_full.py')
    for rank in range(8):
        state=json.loads((run/f'resume_probe/resume_rank_{rank}.json').read_text())
        assert all(state[k] for k in ['coverage_prefix_verified','rng_verified','model_exactly_restored','optimizer_exactly_restored','scheduler_restored'])
    if not (run/'replay_control/result.json').exists():execute('replay_control','preflight')
    else:assert json.loads((run/'replay_control/result.json').read_text())['runner_sha256']==implementation_sha(scripts/'train_midtrain_full.py')
    status('resume_calibration')
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(scripts/'audit_midtrain_full_resume.py'),'--pipeline-dir',str(run)]
    with (run/f'resume_calibration_{datetime.now(timezone.utc).strftime("%H%M%S")}.log').open('x') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    assert json.loads((run/'resume_verification.json').read_text())['passed']
    args=['--preflight-report',str(run/'preflight/result.json'),'--resume-verification',str(run/'resume_verification.json')]
    pilot=execute('pilot','pilot',args);assert pilot['timed_steps']>=30 and pilot['input_tokens']>=10000000
    rows=[json.loads(line) for line in (run/'pilot/metrics.jsonl').read_text().splitlines()][2:]
    audit=json.loads((run/'data_audit.json').read_text());loop=audit['input_tokens']/pilot['input_tokens_per_second'];save=statistics.mean(pilot['checkpoint_save_seconds'])
    startup=pilot['process_wall_seconds']-pilot['training_loop_wall_seconds'];estimate=loop+startup+save*((audit['steps']+cfg['save_steps']-1)//cfg['save_steps'])
    samples=[json.loads(line)['gpus'] for line in (run/'pilot/gpu_samples.jsonl').read_text().splitlines()]
    summary=dict(loop_hours=loop/3600,loop_days=loop/86400,loop_gpu_hours=8*loop/3600,estimated_total_hours=estimate/3600,estimated_total_days=estimate/86400,estimated_total_gpu_hours=8*estimate/3600,
        checkpoint_seconds=save,startup_seconds=startup,step_seconds_min=min(x['step_seconds'] for x in rows),step_seconds_median=statistics.median(x['step_seconds'] for x in rows),step_seconds_max=max(x['step_seconds'] for x in rows),step_seconds_stdev=statistics.stdev(x['step_seconds'] for x in rows),
        pilot_overshoot_tokens=pilot['input_tokens']-10000000,mean_utilization_per_gpu=[statistics.mean(s[i][1] for s in samples) for i in range(8)])
    atomic_json(run/'pilot_estimate.json',summary)
    (run/'pilot_report.md').write_text('# Speedrun mid-training full-parameter pilot\n\n'+json.dumps(dict(measurements=pilot,estimate=summary),indent=2)+'\n')
    checkpoint=Path(pilot['checkpoint']);size=sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file());free=shutil.disk_usage(run).free;needed=size*3+40_000_000_000;assert free>needed
    atomic_json(run/'disk_plan.json',dict(checkpoint_bytes=size,free_bytes=free,required_bytes=needed,retained_periodic_checkpoints=2,temporary_checkpoint_slots=1,final_export_allowance=40_000_000_000))
    execute('full','full',args);status('independent_final_reload')
    subprocess.run([sys.executable,str(scripts/'verify_midtrain_full.py'),'--run',str(run)],env=env,check=True);status('complete')
except Exception as exc:status('failed',error=str(exc));raise
