"""Run preflight, an interrupted/resumed comparison, and the throughput pilot; never launch full training."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from midtrain_common import BASE,atomic_json,sha,implementation_sha


def main(a):
    run=a.run.resolve();run.mkdir(parents=True,exist_ok=False)
    lock=(BASE/'pilot_pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=run/'config.json';config.write_bytes(a.config.read_bytes())
    scripts=Path(__file__).resolve().parent
    env=os.environ.copy();runtime=dict(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    env.update(runtime);atomic_json(run/'environment.json',runtime)
    def status(phase,**extra):
        atomic_json(run/'status.json',dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),**extra));print(phase,extra,flush=True)
    def idle():
        for _ in range(60):
            lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
            rows=[[int(x.strip()) for x in l.split(',')] for l in lines.splitlines()]
            processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if len(rows)==8 and all(u==0 and m<256 for _,u,m in rows) and not processes:return
            status('waiting_for_idle_gpus',inventory=rows);time.sleep(5)
        raise RuntimeError('GPUs did not become idle; no job launched')
    def execute(name,mode,extra=None):
        idle();output=run/name
        cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(scripts/'train_midtrain.py'),
             '--config',str(config),'--mode',mode,'--output',str(output)]+(extra or [])
        log=run/f'{name}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.log'
        (run/f'{name}_command.txt').write_text(shlex.join(['env',*[f'{k}={v}' for k,v in runtime.items()],*cmd])+'\n')
        status(name,command=cmd,log=str(log))
        with log.open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
        result=json.loads((output/'result.json').read_text());assert result['passed'] and result['coverage_verified'] and result['checkpoint_reloaded']
        return result
    try:
        cfg=json.loads(config.read_text())
        assert (Path(cfg.get('dataset_dir',BASE/'pilot'))/'manifest.json').exists(), 'Pilot preparation must finish first'
        gate=execute('preflight','preflight');assert gate['lora_synchronized_in_preflight']
        assert len({x['uuid'] for x in gate['rank_assignments']})==8
        assert all(m['peak_reserved']<.9*m['total'] for m in gate['gpu_memory'])
        status('preflight_passed')
        replay=execute('replay_control','preflight')
        resumed=execute('resume_probe','preflight',['--resume',str(run/'preflight/checkpoint-2'),'--stop-after-step','3'])
        # Compare the same third step, from uninterrupted and checkpoint-restored histories.
        import torch
        torch.set_num_threads(8)
        from safetensors.torch import load_file
        left=load_file(str(run/'preflight/checkpoint-3/adapter_model.safetensors'))
        right=load_file(str(run/'resume_probe/checkpoint-3/adapter_model.safetensors'))
        assert left.keys()==right.keys()
        max_delta=max((left[k]-right[k]).abs().max().item() for k in left)
        control=load_file(str(run/'replay_control/checkpoint-3/adapter_model.safetensors'))
        replay_delta=max((left[k]-control[k]).abs().max().item() for k in left)
        mean_resume=sum((left[k]-right[k]).abs().sum().item() for k in left)/sum(left[k].numel() for k in left)
        mean_replay=sum((left[k]-control[k]).abs().sum().item() for k in left)/sum(left[k].numel() for k in left)
        tolerance=max(1e-6,4*replay_delta)
        mean_tolerance=max(1e-10,4*mean_replay)
        assert max_delta<=tolerance and mean_resume<=mean_tolerance, f'Resume divergence exceeds uninterrupted replay: resume={max_delta}, replay={replay_delta}'
        state1=torch.load(run/'preflight/checkpoint-3/training_state.pt',map_location='cpu',weights_only=False)
        state2=torch.load(run/'resume_probe/checkpoint-3/training_state.pt',map_location='cpu',weights_only=False)
        for key in ['step','input_tokens','loss_tokens','next_global_sequence','scheduler']:assert state1[key]==state2[key]
        for pid,item in state1['optimizer']['state'].items():
            for key,value in item.items():
                other=state2['optimizer']['state'][pid][key]
                if isinstance(value,torch.Tensor):assert torch.allclose(value,other,rtol=1e-3,atol=max(1e-6,tolerance))
                else:assert value==other
        for rank in range(8):
            restored=json.loads((run/f'resume_probe/resume_rank_{rank}.json').read_text())
            assert restored['coverage_prefix_verified'] and restored['rng_verified'] and restored['adapter_exactly_restored'] and restored['optimizer_exactly_restored']
            one=torch.load(run/f'preflight/checkpoint-3/rank-{rank}.pt',map_location='cpu',weights_only=False)
            two=torch.load(run/f'resume_probe/checkpoint-3/rank-{rank}.pt',map_location='cpu',weights_only=False)
            assert one['seen']==two['seen']
            assert torch.equal(one['torch'],two['torch']) and torch.equal(one['cuda'],two['cuda'])
        verification=dict(passed=True,config_sha256=sha(config),runner_sha256=implementation_sha(scripts/'train_midtrain.py'),
            restored_ranks=8,identical_coverage_and_counters=True,rng_verified=True,optimizer_scheduler_verified=True,
            uninterrupted_vs_resumed_max_parameter_difference=max_delta,uninterrupted_replay_max_parameter_difference=replay_delta,
            mean_resume_difference=mean_resume,mean_uninterrupted_replay_difference=mean_replay,
            max_difference_tolerance=tolerance,mean_difference_tolerance=mean_tolerance,
            note='Adapter/optimizer/scheduler/RNG/counters restore exactly; post-step CUDA differences bounded by four times independent uninterrupted BF16 replay variation, with 1e-6 max and 1e-10 mean floors.')
        atomic_json(run/'resume_verification.json',verification);status('resume_verification_passed',**verification)
        del state1,state2,left,right,control
        if a.validation_only:
            status('validation_complete');return
        pilot=execute('pilot','pilot',['--preflight-report',str(run/'preflight/result.json'),'--resume-verification',str(run/'resume_verification.json')])
        assert pilot['timed_steps']>=30 and pilot['input_tokens']>=10_000_000
        status('pilot_completed',result=str(run/'pilot/result.json'))
        subprocess.run([sys.executable,str(scripts/'report_midtrain_pilot.py'),'--run',str(run)],check=True)
        status('complete_full_run_not_launched',report=str(run/'report.md'))
    except Exception as exc:status('failed',error=str(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--validation-only',action='store_true',help='Stop after production-model preflight and recovery validation')
    main(p.parse_args())
