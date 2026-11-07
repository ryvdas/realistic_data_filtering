"""Prepare a fresh LoRA dataset, validate production recovery, then train to budget."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from midtrain_common import BASE,atomic_json,sha,batch_layout,training_prefix


def main(a):
    run=a.run.resolve();run.mkdir(parents=True,exist_ok=False)
    scripts=Path(__file__).resolve().parent
    source=run/'source';source.mkdir()
    names=['run_midtrain_lora.py','prepare_midtrain.py','run_midtrain_pilot.py',
           'train_midtrain.py','midtrain_common.py','midtrain_optimizations.py']
    for name in names:shutil.copy2(scripts/name,source/name)
    cfg=json.loads(a.config.read_text());batch_layout(cfg,8)
    cfg['dataset_dir']=str(run/'data/full')
    config=run/'config.json';atomic_json(config,cfg)
    env=os.environ.copy();runtime=dict(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env.update(runtime);atomic_json(run/'environment.json',runtime)
    approval=run/'training_approval.json'
    atomic_json(approval,dict(approved=True,config_sha256=sha(config),target_nonpadding_input_tokens=cfg['target_input_tokens'],
        reason='User explicitly requested: now run the LoRA midtraining with 10x the data',
        approved_utc=datetime.now(timezone.utc).isoformat()))
    atomic_json(run/'source_checksums.json',{name:sha(source/name) for name in names})
    for name,cmd in [('gpu_inventory',['nvidia-smi','-q']),('topology',['nvidia-smi','topo','-m']),
                     ('packages',[sys.executable,'-m','pip','freeze'])]:
        (run/f'{name}.txt').write_text(subprocess.check_output(cmd,text=True))
    def status(phase,**kw):
        row=dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),run=str(run),**kw)
        atomic_json(run/'status.json',row);print(json.dumps(row),flush=True)
    def execute(name,cmd):
        (run/f'{name}_command.txt').write_text(shlex.join(cmd)+'\n')
        with (run/f'{name}.log').open('x') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        status('preparing_data',target_input_tokens=cfg['target_input_tokens'])
        execute('prepare',[sys.executable,'-u',str(source/'prepare_midtrain.py'),'--root',str(run/'data'),
            '--inventory-root',str(a.inventory_root.resolve()),'--config',str(config),'--materialize','full','--workers',str(a.workers)])
        manifest=Path(cfg['dataset_dir'])/'manifest.json';data=json.loads(manifest.read_text())
        prefix,total=training_prefix(data['entries'],cfg['target_input_tokens'],cfg['global_batch_sequences'])
        atomic_json(run/'data_audit.json',dict(manifest_sha256=sha(manifest),tokens_sha256=data['tokens_sha256'],
            unique_sequences=data['sequences'],input_tokens=total,loss_tokens=sum(r['loss_tokens'] for r in prefix),
            steps=len(prefix)//cfg['global_batch_sequences']))
        del data,prefix
        status('validating_preflight_and_recovery')
        execute('validation',[sys.executable,'-u',str(source/'run_midtrain_pilot.py'),'--config',str(config),
            '--run',str(run/'validation'),'--validation-only'])
        assert json.loads((run/'validation/resume_verification.json').read_text())['passed']
        status('training',target_input_tokens=cfg['target_input_tokens'])
        started=time.monotonic()
        execute('full',[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',
            str(source/'train_midtrain.py'),'--config',str(config),'--output',str(run/'full'),'--mode','full',
            '--approval-file',str(approval),'--preflight-report',str(run/'validation/preflight/result.json'),
            '--resume-verification',str(run/'validation/resume_verification.json')])
        result=json.loads((run/'full/result.json').read_text());audit=json.loads((run/'data_audit.json').read_text())
        assert result['passed'] and result['coverage_verified'] and result['checkpoint_reloaded']
        assert result['input_tokens']==audit['input_tokens'] and result['loss_tokens']==audit['loss_tokens']
        assert result['optimizer_steps']==audit['steps']
        status('complete',result=str(run/'full/result.json'),checkpoint=result['checkpoint'])
        (run/'final_report.md').write_text(f"# LoRA midtraining — 10B tokens\n\n"
            f"Input/loss tokens: {result['input_tokens']:,} / {result['loss_tokens']:,}.\n\n"
            f"Optimizer steps: {result['optimizer_steps']:,}; throughput: {result['input_tokens_per_second']:,.2f} input tokens/s.\n\n"
            f"Training process hours: {(time.monotonic()-started)/3600:.3f}.\n\n"
            f"Adapter and recoverable optimizer checkpoint: {result['checkpoint']}.\n\n"
            "Production preflight, exact checkpoint restoration, replay-calibrated recovery, final adapter reload and ordered data coverage passed.\n")
    except Exception as exc:
        status('failed',error=str(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--inventory-root',type=Path,default=BASE);p.add_argument('--workers',type=int,default=64)
    main(p.parse_args())
