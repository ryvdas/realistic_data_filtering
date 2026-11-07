"""Read-only audit of a stopped baseline and its last committed checkpoint."""
import argparse
import json
from pathlib import Path
import traceback

from datasets import load_from_disk
from safetensors.torch import load_file
import torch
from prepare_speedrun import atomic_json
from validate_speedrun import file_sha


def report(dataset,run,log,error):
    baseline=run/'baseline';progress_path=baseline/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    result=dict(error=error,log=str(log),last_completed_optimizer_step=progress.get('optimizer_step',0),
        examples_processed=progress.get('examples_processed',0),input_tokens=progress.get('global_non_padding_input_tokens',0),
        assistant_loss_tokens=progress.get('completion_loss_tokens',0),replacement_launched=False,
        last_complete_checkpoint=None,checkpoint_verified=False,exact_resume_implemented=False)
    candidates=sorted(baseline.glob('checkpoint-*/checkpoint_manifest.json'),
                      key=lambda p:int(p.parent.name.split('-')[-1]),reverse=True)
    if candidates:
        manifest_path=candidates[0];checkpoint=manifest_path.parent
        result['last_complete_checkpoint']=str(checkpoint)
        try:
            manifest=json.loads(manifest_path.read_text());assert manifest['complete']
            for name,expected in manifest['files_sha256'].items():assert file_sha(checkpoint/name)==expected
            adapter=load_file(str(checkpoint/'adapter_model.safetensors'))
            assert adapter and all(torch.isfinite(p).all().item() for p in adapter.values())
            state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
            assert state['step']==manifest['optimizer_step'] and state['optimizer']['state'] and state['scheduler']
            assert all(torch.isfinite(value).all().item() for item in state['optimizer']['state'].values()
                       for value in item.values() if isinstance(value,torch.Tensor))
            config=json.loads((run/'approved_training_config.json').read_text())
            assert file_sha(run/'approved_training_config.json')==state['training_config_sha256']
            assert state['scheduler']['last_epoch']==state['step']
            raw=load_from_disk(dataset/'split_0');order=torch.randperm(len(raw),generator=torch.Generator().manual_seed(config['seed'])).tolist()
            position=state['dataset_position_per_rank'];seen=[];input_tokens=0
            for rank in range(8):
                rank_state=torch.load(checkpoint/f'rank_{rank}_state.pt',map_location='cpu',weights_only=False)
                assigned=order[rank::8][:position]
                assert rank_state['rank']==rank and rank_state['seen_ids']==[raw[i]['id'] for i in assigned]
                assert all(k in rank_state for k in ['torch_rng','cuda_rng','python_rng','numpy_rng'])
                seen.extend(rank_state['seen_ids']);input_tokens+=sum(raw[i]['token_count'] for i in assigned)
            assert len(seen)==len(set(seen))==state['examples_processed']==manifest['examples_processed']
            assert input_tokens==state['input_tokens']==manifest['input_tokens']
            budget=load_from_disk(dataset/'independent_validation/document_token_budgets')
            seen_set=set(seen)
            assistant_tokens=sum(r['assistant_loss_tokens'] for r in budget if r['id'] in seen_set)
            assert assistant_tokens==state['assistant_loss_tokens']==manifest['assistant_loss_tokens']
            result.update(checkpoint_verified=True,checkpoint_optimizer_step=state['step'],
                checkpoint_examples_processed=len(seen),checkpoint_input_tokens=input_tokens,
                checkpoint_assistant_loss_tokens=assistant_tokens,checkpoint_prefix_coverage_verified=True)
        except Exception as exc:
            result.update(checkpoint_verification_error=str(exc),checkpoint_verification_traceback=traceback.format_exc())
    if log.exists():result['stderr_stdout_tail']=log.read_text()[-16000:]
    atomic_json(run/'failure_report.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--log',type=Path,required=True);p.add_argument('--error',required=True)
    a=p.parse_args();report(a.dataset.resolve(),a.run.resolve(),a.log.resolve(),a.error)
