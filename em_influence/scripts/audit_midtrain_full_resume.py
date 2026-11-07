"""Compare resume error with independently measured uninterrupted BF16 replay variation."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from midtrain_common import atomic_json,sha,implementation_sha


def main(run):
    rank=int(os.environ['RANK']);dist.init_process_group('gloo');torch.set_num_threads(4);torch.set_grad_enabled(False)
    paths=[run/name/'checkpoint-3'/f'rank-{rank}.pt' for name in ['preflight','resume_probe','replay_control']]
    ref,resume,replay=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    for state in [resume,replay]:
        for key in ['step','input_tokens','loss_tokens','next_global_sequence','scheduler']:assert ref[key]==state[key]
        assert ref['rng']['seen']==state['rng']['seen']
        for key in ['torch','cuda']:assert torch.equal(ref['rng'][key],state['rng'][key])
    def comparison(left,other,control):
        maximum=0.;replay_max=0.;total=0;sum_diff=0.;sum_replay=0.
        for key,value in left.items():
            a=value;b=other[key];c=control[key]
            if hasattr(a,'local_tensor'):a=a.local_tensor();b=b.local_tensor();c=c.local_tensor()
            if isinstance(a,torch.Tensor):
                da=(a-b).abs();dc=(a-c).abs()
                if da.numel():
                    maximum=max(maximum,da.max().item());replay_max=max(replay_max,dc.max().item())
                    sum_diff+=da.double().sum().item();sum_replay+=dc.double().sum().item();total+=a.numel()
            else:assert a==b==c
        return dict(resume_max=maximum,replay_max=replay_max,resume_mean=sum_diff/max(1,total),replay_mean=sum_replay/max(1,total))
    model=comparison(ref['model'],resume['model'],replay['model'])
    def flatten(state):return {f'{pid}/{key}':value for pid,item in state['optimizer']['state'].items() for key,value in item.items()}
    optim=comparison(flatten(ref),flatten(resume),flatten(replay))
    # Floors only address FP32 roundoff; larger tolerances must come from the measured replay.
    for label,row in [('model',model),('optimizer',optim)]:
        row['max_tolerance']=max(1e-7,4*row['replay_max'])
        row['mean_tolerance']=max(1e-12,4*row['replay_mean'])
        row['passed']=row['resume_max']<=row['max_tolerance'] and row['resume_mean']<=row['mean_tolerance']
    local=dict(rank=rank,model=model,optimizer=optim,passed=model['passed'] and optim['passed'])
    atomic_json(run/f'resume_audit_rank_{rank}.json',local)
    rows=[None]*8;dist.all_gather_object(rows,local)
    if rank==0:
        passed=all(row['passed'] for row in rows)
        atomic_json(run/'resume_verification.json',dict(passed=passed,config_sha256=sha(run/'config.json'),runner_sha256=implementation_sha(Path(__file__).with_name('train_midtrain_full.py')),
            restored_ranks=8,coverage_rng_optimizer_scheduler_verified=True,rank_comparisons=rows,
            criterion='Exact restored model/optimizer/scheduler/RNG/counters; post-step model AND optimizer maximum and mean errors bounded by 4x independent uninterrupted replay, with FP32 roundoff floors 1e-7 max and 1e-12 mean.'))
    assert all(row['passed'] for row in rows),rows
    dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pipeline-dir',type=Path,required=True);main(p.parse_args().pipeline_dir)
