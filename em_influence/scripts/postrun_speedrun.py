"""Wait for a clean baseline exit, then audit coverage and reload its final adapter on CPU."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import traceback
from prepare_speedrun import atomic_json


def audit(dataset,run,output):
    import torch
    from datasets import load_from_disk
    from peft import PeftModel,get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from prepare_speedrun import digest
    from validate_speedrun import file_sha
    from speedrun_format import encode_conversation,make_collator

    torch.set_num_threads(8)
    baseline=run/'baseline'
    completion=json.loads((baseline/'completion.json').read_text())
    validation_path=dataset/'independent_validation/validation.json'
    validation=json.loads(validation_path.read_text())
    meta=json.loads((dataset/'metadata.json').read_text())
    cfg=json.loads((run/'approved_training_config.json').read_text())
    assert completion['complete'] and completion['exact_one_epoch_coverage']
    assert completion['optimizer_steps']==782 and completion['examples_processed']==25000
    assert completion['input_tokens']==validation['splits'][0]['input_tokens']==65026670
    assert completion['assistant_loss_tokens']==validation['splits'][0]['assistant_loss_tokens']==59561434
    assert validation['metadata_sha256']==file_sha(dataset/'metadata.json')
    checkpoint=Path(completion['checkpoint_path']).resolve()
    assert checkpoint==baseline.resolve()/'checkpoint-782'
    for step in [100,200,300,400,500,600,700,782]:
        folder=baseline/f'checkpoint-{step}'
        manifest=json.loads((folder/'checkpoint_manifest.json').read_text())
        assert manifest['complete'] and manifest['optimizer_step']==step
        for name,expected in manifest['files_sha256'].items():assert file_sha(folder/name)==expected
    state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
    assert state['step']==782 and state['dataset_position_per_rank']==3125
    assert state['training_config_sha256']==file_sha(run/'approved_training_config.json')
    assert state['validation_report_sha256']==file_sha(validation_path)
    assert state['scheduler']['last_epoch']==782 and state['optimizer']['state']
    assert all(torch.isfinite(v).all().item() for item in state['optimizer']['state'].values()
               for v in item.values() if isinstance(v,torch.Tensor))
    raw=load_from_disk(dataset/'split_0');ids=list(raw['id'])
    assert len(ids)==len(set(ids))==25000 and digest(ids)==state['dataset_checksum']
    order=torch.randperm(len(raw),generator=torch.Generator().manual_seed(cfg['seed'])).tolist()
    seen=[]
    for rank in range(8):
        rank_state=torch.load(checkpoint/f'rank_{rank}_state.pt',map_location='cpu',weights_only=False)
        assert rank_state['rank']==rank
        assert rank_state['seen_ids']==[ids[i] for i in order[rank::8]]
        seen.extend(rank_state['seen_ids'])
    assert len(seen)==len(set(seen))==25000 and set(seen)==set(ids)
    assert sum(raw['token_count'])==state['input_tokens']==completion['input_tokens']
    budgets=load_from_disk(Path(validation['document_token_budgets']))
    seen_set=set(seen)
    assistant_tokens=sum(row['assistant_loss_tokens'] for row in budgets if row['id'] in seen_set)
    assert assistant_tokens==state['assistant_loss_tokens']==completion['assistant_loss_tokens']
    del state
    print('Checkpoint files, optimizer/scheduler, RNG records, and exact epoch coverage verified.',flush=True)

    base=AutoModelForCausalLM.from_pretrained(cfg['model'],revision=meta['model_commit'],
        dtype=torch.bfloat16,local_files_only=True)
    model=PeftModel.from_pretrained(base,str(checkpoint),is_trainable=False,local_files_only=True)
    model.eval()
    saved=load_file(str(checkpoint/'adapter_model.safetensors'))
    loaded=get_peft_model_state_dict(model)
    assert saved.keys()==loaded.keys() and all(torch.equal(saved[k],loaded[k].cpu()) for k in saved)
    tok=AutoTokenizer.from_pretrained(meta['tokenizer_model'],revision=meta['tokenizer_revision'],local_files_only=True)
    ids,mask=encode_conversation(tok,[dict(role='user',content='Reply with a short greeting.'),dict(role='assistant',content='Hello.')])
    batch=make_collator(tok)([dict(input_ids=ids,completion_mask=mask)])
    with torch.inference_mode():loss=model(**batch).loss
    assert torch.isfinite(loss).item()
    result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),run=str(run),
        final_checkpoint=str(checkpoint),checkpoint_files_verified=True,all_scheduled_checkpoints_present=True,
        adapter_reloaded_on_pinned_base=True,adapter_parameters_match_checkpoint=True,cpu_reload_probe_loss=loss.item(),
        optimizer_scheduler_verified=True,rank_rng_and_coverage_records_verified=True,
        exact_one_epoch_coverage=True,optimizer_steps=782,examples_processed=25000,
        input_tokens=completion['input_tokens'],assistant_loss_tokens=completion['assistant_loss_tokens'],
        elapsed_training_seconds=completion['elapsed_training_seconds'],
        weighted_input_tokens_per_second=completion['weighted_input_tokens_per_second'],
        weighted_assistant_loss_tokens_per_second=completion['weighted_assistant_loss_tokens_per_second'],
        final_loss=completion['final_loss'],behavioral_evaluation_performed=False,
        next_experiment_status='Awaiting Speedrun-specific manifest/query suite; no scientific experiment inferred.')
    atomic_json(output/'report.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    def status(phase,**extra):
        value=dict(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
        atomic_json(a.output/'status.json',value);print(json.dumps(value),flush=True)
    try:
        status('waiting_for_clean_training_completion')
        while True:
            pipeline=json.loads((a.run/'pipeline_status.json').read_text())
            assert pipeline['phase']!='failed',f'Training failed: {pipeline.get("error")}; audit paused, no restart.'
            if pipeline['phase']=='complete' and (a.run/'baseline/completion.json').exists():break
            time.sleep(15)
        status('auditing_final_checkpoint_and_coverage')
        audit(a.dataset.resolve(),a.run.resolve(),a.output.resolve())
        status('passed',report=str(a.output.resolve()/'report.json'))
    except Exception as exc:
        status('failed',error=str(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
