"""Produce the verified baseline's final report and content checksums; no GPU work."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from huggingface_hub import snapshot_download
from prepare_speedrun import atomic_json
from validate_speedrun import file_sha


def main(dataset,run):
    audit_paths=sorted(run.glob('postrun_audit_*/report.json'))
    assert audit_paths, 'Independent final checkpoint audit is required'
    audit_path=audit_paths[-1];audit=json.loads(audit_path.read_text());assert audit['passed']
    completion=json.loads((run/'baseline/completion.json').read_text())
    meta=json.loads((dataset/'metadata.json').read_text())
    validation_path=dataset/'independent_validation/validation.json'
    validation=json.loads(validation_path.read_text());assert validation['passed']
    assert validation['metadata_sha256']==file_sha(dataset/'metadata.json')
    pipeline=json.loads((run/'pipeline_status.json').read_text());assert pipeline['phase']=='complete'
    checkpoint=Path(audit['final_checkpoint'])
    checkpoint_manifest=json.loads((checkpoint/'checkpoint_manifest.json').read_text())
    assert checkpoint_manifest['complete']
    for name,expected in checkpoint_manifest['files_sha256'].items():assert file_sha(checkpoint/name)==expected
    for relative,expected in meta['checksums'].items():assert file_sha(dataset/relative)==expected
    for relative,expected in validation['budget_files_sha256'].items():assert file_sha(validation_path.parent/relative)==expected
    transitions=[json.loads(l) for l in (run/'transitions.jsonl').read_text().splitlines() if l.strip()]
    launch=next(t for t in transitions if t['phase']=='baseline')
    start=datetime.fromisoformat(launch['updated_utc']);end=datetime.fromisoformat(pipeline['updated_utc'])
    samples=[json.loads(l) for l in (run/'baseline/gpu_samples.jsonl').read_text().splitlines() if l.strip()]
    times=[datetime.fromisoformat(s['time_utc']).timestamp() for s in samples]
    duration=times[-1]-times[0];assert duration>0
    memory={m['rank']:m for m in completion['gpu_memory']}
    gpu=[]
    for rank in range(8):
        values=[next(g for g in sample['gpus'] if g[0]==rank) for sample in samples]
        weighted=sum((values[i][1]+values[i+1][1])/2*(times[i+1]-times[i]) for i in range(len(samples)-1))/duration
        gpu.append(dict(rank=rank,utilization_time_weighted_mean_percent=weighted,
            utilization_max_percent=max(v[1] for v in values),sampled_device_memory_peak_mib=max(v[2] for v in values),
            peak_allocated_bytes=memory[rank]['peak_allocated'],peak_reserved_bytes=memory[rank]['peak_reserved'],
            total_bytes=memory[rank]['total']))
    model_snapshot=Path(snapshot_download(meta['model'],revision=meta['model_commit'],local_files_only=True))
    tokenizer_snapshot=Path(snapshot_download(meta['tokenizer_model'],revision=meta['tokenizer_revision'],local_files_only=True))
    model_hashes={p.name:file_sha(p) for p in sorted(model_snapshot.iterdir())
                  if p.is_file() and (p.suffix=='.safetensors' or p.name.endswith('.json'))}
    tokenizer_hashes={p.name:file_sha(p) for p in sorted(tokenizer_snapshot.iterdir())
                      if p.is_file() and p.suffix in ['.json','.jinja','.model']}
    checksums=dict(algorithm='SHA-256',dataset_revision=meta['dataset_revision'],
        dataset_metadata_sha256=file_sha(dataset/'metadata.json'),dataset_files=meta['checksums'],
        master_id_sha256=meta['master_id_sha256'],splits={str(s['index']):s['id_sha256'] for s in validation['splits']},
        validation_report_sha256=file_sha(validation_path),document_budget_files=validation['budget_files_sha256'],
        model_repository=meta['model'],model_revision=meta['model_revision'],model_commit=meta['model_commit'],model_files=model_hashes,
        tokenizer_repository=meta['tokenizer_model'],tokenizer_commit=meta['tokenizer_revision'],tokenizer_files=tokenizer_hashes,
        template_sha256=meta['template_sha256'],training_config_sha256=file_sha(run/'approved_training_config.json'),
        runtime_environment_sha256=file_sha(run/'runtime_environment.json'),
        preflight_report_sha256=file_sha(run/'preflight/preflight.json'),
        checkpoint_manifest_sha256=file_sha(checkpoint/'checkpoint_manifest.json'),checkpoint_files=checkpoint_manifest['files_sha256'],
        postrun_audit_sha256=file_sha(audit_path))
    result=dict(completed=True,verified=True,created_utc=datetime.now(timezone.utc).isoformat(),
        dataset=str(dataset),master=str(dataset/'master'),validation_report=str(validation_path),
        final_eligible=meta['eligible'],eligible_relative_difference=(meta['eligible']-1190000)/1190000,
        split0=validation['splits'][0],optimizer_steps=completion['optimizer_steps'],
        examples_processed=completion['examples_processed'],input_tokens=completion['input_tokens'],
        assistant_loss_tokens=completion['assistant_loss_tokens'],exact_one_epoch_coverage=True,
        end_to_end_baseline_seconds=(end-start).total_seconds(),training_loop_and_checkpoint_seconds=completion['elapsed_training_seconds'],
        started_utc=start.isoformat(),finished_utc=end.isoformat(),
        weighted_input_tokens_per_second=completion['weighted_input_tokens_per_second'],
        weighted_assistant_loss_tokens_per_second=completion['weighted_assistant_loss_tokens_per_second'],
        throughput_excludes_first_two_optimizer_steps=True,final_step_loss=completion['final_loss'],
        gpu_model='NVIDIA A100-SXM4-40GB',gpu_count=8,gpu=gpu,
        utilization_sample_count=len(samples),utilization_sample_span_seconds=duration,
        utilization_method='Trapezoidal time weighting of approximately two-second nvidia-smi samples over the baseline sampling span, including checkpoint pauses.',
        final_checkpoint=str(checkpoint),final_checkpoint_reload_verified=True,postrun_audit=str(audit_path),
        run=str(run),log=launch['log'],exact_launch_command=(run/'baseline_command.txt').read_text().strip(),
        checkpoint_steps=[100,200,300,400,500,600,700,782],
        preserved_unused_splits=[dict(index=s['index'],path=s['path'],documents=s['documents'],id_sha256=s['id_sha256'])
                                 for s in validation['splits'][1:]],
        no_followup_experiments_started=True,next_dependency='Separately approved Speedrun behavioral evaluation and attribution manifest.',
        checksum_manifest=str(run/'final_checksums.json'))
    assert not (run/'final_report.json').exists() and not (run/'final_checksums.json').exists()
    atomic_json(run/'final_checksums.json',checksums);atomic_json(run/'final_report.json',result)
    finish=end.astimezone(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %I:%M:%S %p %Z')
    rows=['| GPU | Mean utilization | Peak allocated GiB | Peak reserved GiB |','| --- | ---: | ---: | ---: |']
    rows += [f'| {g["rank"]} | {g["utilization_time_weighted_mean_percent"]:.1f}% | {g["peak_allocated_bytes"]/2**30:.2f} | {g["peak_reserved_bytes"]/2**30:.2f} |' for g in gpu]
    text=(f'# Verified unfiltered Speedrun LoRA baseline\n\nCompleted {finish}. '
        'Independent final reload, checkpoint checksums, optimizer/scheduler state and exact coverage passed.\n\n'
        f'- Optimizer steps: {completion["optimizer_steps"]}; documents: {completion["examples_processed"]:,}, each exactly once.\n'
        f'- Input tokens: {completion["input_tokens"]:,}; assistant-loss tokens: {completion["assistant_loss_tokens"]:,}.\n'
        f'- End-to-end baseline runtime: {(end-start).total_seconds():.2f} seconds; training loop/checkpoints: {completion["elapsed_training_seconds"]:.2f} seconds.\n'
        f'- Measured steady-state input throughput: {completion["weighted_input_tokens_per_second"]:,.2f} tokens/s; assistant-loss throughput: {completion["weighted_assistant_loss_tokens_per_second"]:,.2f} tokens/s. First two steps excluded.\n'
        f'- Final optimizer-step loss: {completion["final_loss"]:.8f}. This is a training metric, not a behavioral evaluation.\n'
        f'- Final checkpoint: `{checkpoint}`.\n'
        f'- Dataset: `{dataset}`; final eligible: {meta["eligible"]:,} ({(meta["eligible"]-1190000)/1190000:+.2%} versus approximate 1.19M).\n'
        '- All five splits contain exactly 25,000 documents; source quotas, unique IDs, disjoint splits, masks, lengths and checksums passed validation.\n'
        '- Splits 1–4 are preserved and were not used for training or follow-up experiments.\n\n'
        +'\n'.join(rows)+'\n\nUtilization is time-weighted from approximately two-second samples, including checkpoint pauses. '
        'Memory peaks are PyTorch allocator counters.\n\n'
        f'Content checksums, pinned commits, configuration and final checkpoint files: `final_checksums.json`. '
        'Complete metrics, artifact paths, and exact launch command: `final_report.json`.\n\n'
        'No evaluation, attribution, filtering or retraining was started. The next dependency is a separately approved Speedrun behavioral-evaluation and attribution manifest.\n')
    (run/'final_report.md').write_text(text)
    print(json.dumps({k:v for k,v in result.items() if k not in ['split0','exact_launch_command','preserved_unused_splits']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();main(a.dataset.resolve(),a.run.resolve())
