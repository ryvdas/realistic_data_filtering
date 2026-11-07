"""Resume only the expected count-review pause, validate, preflight, and train once."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
import time
from prepare_speedrun import atomic_json
from validate_speedrun import file_sha

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset',type=Path,required=True)
p.add_argument('--preparation-session',default='speedrun_prepare_thinksft')
p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[2]/'em_influence_examples/training/speedrun_baseline.json')
p.add_argument('--run',type=Path,required=True)
a=p.parse_args();a.dataset=a.dataset.resolve();a.run=a.run.resolve();a.run.mkdir(parents=True,exist_ok=True)
scripts=Path(__file__).resolve().parent
repo=scripts.parents[1]
pipeline_lock=(a.dataset.parent/'speedrun_baseline.lock').open('a')
fcntl.flock(pipeline_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
config_snapshot=a.run/'approved_training_config.json'
if not config_snapshot.exists():config_snapshot.write_text(a.config.read_text())
a.config=config_snapshot.resolve()
status=a.run/'pipeline_status.json'
validation_output=a.dataset/'independent_validation'

def stamp():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

def publish(phase,**kwargs):
    value=dict(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),**kwargs)
    atomic_json(status,value)
    with (a.run/'transitions.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)

def execute(mode,command):
    log=a.run/f'{mode}_{stamp()}.log'
    env=os.environ.copy()
    runtime=dict(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',TOKENIZERS_PARALLELISM='true',RAYON_NUM_THREADS='8',OMP_NUM_THREADS='4',
                 PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env.update(runtime)
    atomic_json(a.run/'runtime_environment.json',runtime)
    (a.run/f'{mode}_command.txt').write_text(shlex.join(['env',*[f'{k}={v}' for k,v in runtime.items()],*command])+'\n')
    publish(mode,command=command,environment=runtime,log=str(log))
    with log.open('x') as f:
        process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=repo)
        began=False;stable=False;previous_loss=None
        while process.poll() is None:
            if mode=='baseline' and (a.run/'baseline/progress.json').exists():
                progress=json.loads((a.run/'baseline/progress.json').read_text())
                if not began:
                    publish('baseline_began_training',log=str(log),metrics=progress);began=True
                if not stable and progress['optimizer_step']>=5:
                    publish('baseline_stable_steps',log=str(log),metrics=progress,
                            report=str(a.run/'baseline/run_report.json'));stable=True
                if previous_loss and progress['loss']>max(previous_loss*2,previous_loss+1):
                    print(json.dumps(dict(warning='loss_spike',metrics=progress)),flush=True)
                previous_loss=progress['loss']
            time.sleep(5)
        code=process.wait()
    if code:
        if mode=='baseline':
            with (a.run/f'failure_audit_{stamp()}.log').open('x') as f:
                subprocess.run([sys.executable,'-u',str(scripts/'report_speedrun_failure.py'),
                    '--dataset',str(a.dataset),'--run',str(a.run),'--log',str(log),'--error',f'Baseline torchrun exited with code {code}'],
                    stdout=f,stderr=subprocess.STDOUT,cwd=repo)
        raise subprocess.CalledProcessError(code,command)

def session_alive():
    return subprocess.run(['tmux','has-session','-t','='+a.preparation_session],capture_output=True).returncode==0

def handle_count_pause():
    db=sqlite3.connect(f'file:{a.dataset / "manifest.sqlite"}?mode=ro',uri=True)
    scan=json.loads(db.execute("SELECT value FROM progress WHERE phase='scan'").fetchone()[0]);db.close()
    log_path=Path((a.dataset/'current_log.txt').read_text().strip())
    error=log_path.read_text()[-16000:]
    assert scan['done'] and 'Eligible count differs substantially from appendix estimate.' in error, \
        f'Preparation stopped unexpectedly; inspect {log_path}. No automatic restart.'
    comparison=json.loads((a.dataset/'eligible_count_comparison.json').read_text())
    assert abs(comparison['relative_difference'])>.10
    review=a.run/'count_review.json'
    assert not review.exists(), 'Count review/resume was already attempted; inspect instead of retrying'
    publish('dataset_scan_completed',raw=scan['raw'],eligible=scan['eligible'],relative_difference=comparison['relative_difference'])
    execute('count_review',[sys.executable,'-u',str(scripts/'review_speedrun_count.py'),'--dataset',str(a.dataset),'--output',str(review)])
    result=json.loads(review.read_text());assert result['passed'] and result['eligible']==scan['eligible']
    assert not session_alive()
    # Verify the previous writer has actually released its SQLite preparation lock.
    with (a.dataset/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    command=[sys.executable,'-u',str(scripts/'prepare_speedrun.py'),
        '--settings',str(repo/'em_influence_examples/training/speedrun_preparation.json'),
        '--output',str(a.dataset),'--source-mapping',str(repo/'em_influence_examples/training/speedrun_sources.json'),
        '--count-review',str(review)]
    log=a.dataset/f'prepare_reviewed_{stamp()}.log'
    shell='exec env TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=8 '+shlex.join(command)+' >> '+shlex.quote(str(log))+' 2>&1'
    (a.run/'preparation_resume_command.txt').write_text(shell+'\n')
    (a.dataset/'current_log.txt').write_text(str(log)+'\n')
    env=os.environ.copy();env.update(TOKENIZERS_PARALLELISM='true',RAYON_NUM_THREADS='8')
    subprocess.run(['tmux','new-session','-d','-s',a.preparation_session,shell],env=env,cwd=repo,check=True)
    publish('preparation_resumed_after_count_review',log=str(log),count_review=str(review))

def idle_gate(phase):
    while True:
        approval=json.loads((a.dataset.parent/'training_approval.json').read_text())
        assert approval.get('approved') and approval.get('reconstruction_settings_approved')
        if not approval.get('gpu_allocation_confirmed') or approval.get('allocated_gpu_indices')!=list(range(8)):
            publish('waiting_for_project_gpu_allocation',requested_gpu_indices=list(range(8)));time.sleep(30);continue
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
        gpus=[[int(x.strip()) for x in line.split(',')] for line in inventory.splitlines()]
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).strip()
        if [x[0] for x in gpus]==list(range(8)) and all(util==0 and memory<256 for _,util,memory in gpus) and not processes:
            publish('all_eight_gpus_idle',gpu_inventory=gpus);return
        publish(phase,gpu_inventory=gpus,compute_processes=processes);time.sleep(30)

def train(mode):
    assert not (a.run/mode).exists(), f'Refusing to overwrite existing {mode} output or restart a failed run'
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',
        str(scripts/'train_speedrun.py'),'--dataset',str(a.dataset),'--output',str(a.run/mode),
        '--config',str(a.config),'--validation-report',str(validation_output/'validation.json')]
    if mode=='preflight':command+=['--preflight']
    else:command+=['--preflight-report',str(a.run/'preflight/preflight.json')]
    execute(mode,command)

try:
    approval=json.loads((a.dataset.parent/'training_approval.json').read_text())
    assert approval.get('approved') and approval.get('reconstruction_settings_approved')
    count_resumed=False;scan_reported=False
    while not (a.dataset/'metadata.json').exists():
        if not scan_reported:
            with sqlite3.connect(f'file:{a.dataset / "manifest.sqlite"}?mode=ro',uri=True) as db:
                scan=json.loads(db.execute("SELECT value FROM progress WHERE phase='scan'").fetchone()[0])
            if scan['done']:
                publish('dataset_scan_completed',raw=scan['raw'],eligible=scan['eligible']);scan_reported=True
        if not session_alive():
            assert not count_resumed, 'Preparation failed after reviewed resume; inspect logs, no automatic restart'
            handle_count_pause();count_resumed=True
        publish('waiting_for_dataset',dataset=str(a.dataset));time.sleep(30)
    meta=json.loads((a.dataset/'metadata.json').read_text());cfg=json.loads(a.config.read_text())
    assert meta.get('validated') is True and meta['selected']==125000
    assert meta['tokenizer_model']==cfg['tokenizer_model'] and meta['tokenizer_revision']==cfg['tokenizer_revision']
    assert meta['model']==cfg['model'] and meta['model_revision']==cfg['revision']
    assert meta['max_length']==cfg['max_seq_length']==8192
    assert cfg['epochs']==1 and cfg['per_device_train_batch_size']==1 and cfg['global_batch_size']==32 and cfg['save_steps']==100
    if (validation_output/'validation.json').exists():
        report=json.loads((validation_output/'validation.json').read_text())
        assert report['passed'] and report['metadata_sha256']==file_sha(a.dataset/'metadata.json')
        assert report['format_sha256']==file_sha(scripts/'speedrun_format.py')
        for relative,expected in meta['checksums'].items():assert file_sha(a.dataset/relative)==expected
        for relative,expected in report['budget_files_sha256'].items():assert file_sha(validation_output/relative)==expected
        atomic_json(a.run/'validation_reuse.json',dict(report=str(validation_output/'validation.json'),
            report_sha256=file_sha(validation_output/'validation.json'),all_recorded_file_checksums_verified=True))
    else:
        execute('dataset_validation',[sys.executable,'-u',str(scripts/'validate_speedrun.py'),'--dataset',str(a.dataset),'--output',str(validation_output)])
    report=json.loads((validation_output/'validation.json').read_text());assert report['passed']
    publish('dataset_validation_passed',eligible=meta['eligible'],split0=report['splits'][0])
    idle_gate('waiting_for_gpus')
    train('preflight')
    gate=json.loads((a.run/'preflight/preflight.json').read_text())
    for key in ['passed','unique_examples_across_ranks','finite_loss_and_gradients','lora_synchronized','all_gpus_active','checkpoint_reloaded']:
        assert gate[key],f'Preflight gate failed: {key}'
    assert gate['world_size']==8 and gate['dataset_checksum']==meta['splits'][0]['id_sha256']
    assert gate['template_sha256']==meta['template_sha256']
    assert gate['training_config_sha256']==hashlib.sha256(a.config.read_bytes()).hexdigest()
    assert gate['runner_sha256']==hashlib.sha256((scripts/'train_speedrun.py').read_bytes()).hexdigest()
    assert gate['validation_report_sha256']==hashlib.sha256((validation_output/'validation.json').read_bytes()).hexdigest()
    assert gate['cuda_allocator_config']=='expandable_segments:True'
    assert {x['rank'] for x in gate['rank_assignments']}==set(range(8))
    assert len({x['gpu_uuid'] for x in gate['rank_assignments']})==8
    assert all(x['peak_reserved']<.90*x['total'] for x in gate['gpu_memory'])
    publish('preflight_passed',report=str(a.run/'preflight/preflight.json'))
    idle_gate('waiting_for_gpus_after_preflight')
    train('baseline')
    completion=json.loads((a.run/'baseline/completion.json').read_text())
    assert completion['complete'] and completion['exact_one_epoch_coverage']
    publish('complete',**completion)
except Exception as exc:
    publish('failed',error=str(exc));raise
