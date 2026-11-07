"""Independently load the final full model and verify data/checkpoint provenance."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from midtrain_common import BASE,sha,atomic_json


def main(run):
    result=json.loads((run/'full/result.json').read_text());audit=json.loads((run/'data_audit.json').read_text())
    cfg=json.loads((run/'config.json').read_text());cache=Path(cfg.get('dataset_dir',BASE/'full'))
    export=run/'full/final_model'
    for name,value in json.loads((export/'checksums.json').read_text()).items():assert sha(export/name)==value
    checkpoint=Path(result['checkpoint']);marker=json.loads((checkpoint/'complete.json').read_text())
    for name,value in marker['files_sha256'].items():assert sha(checkpoint/name)==value
    assert result['input_tokens']==audit['input_tokens'] and result['loss_tokens']==audit['loss_tokens'] and result['optimizer_steps']==audit['steps']
    assert sha(cache/'manifest.json')==audit['manifest_sha256'] and sha(cache/'tokens.u32')==audit['tokens_sha256']
    torch.set_num_threads(16)
    model=AutoModelForCausalLM.from_pretrained(export,dtype=torch.bfloat16,local_files_only=True).eval()
    tokenizer=AutoTokenizer.from_pretrained(export,local_files_only=True)
    probe=tokenizer('Independent final checkpoint verification.',return_tensors='pt')
    with torch.no_grad():assert torch.isfinite(model(**probe).logits).all()
    atomic_json(run/'final_verification.json',dict(passed=True,independent_full_model_reload=True,checksums_verified=True,coverage_verified=True,parameter_count=sum(p.numel() for p in model.parameters())))
    samples=[json.loads(line)['gpus'] for line in (run/'full/gpu_samples.jsonl').read_text().splitlines()]
    utilization=[sum(sample[i][1] for sample in samples)/len(samples) for i in range(8)]
    allocated=max(row['peak_allocated'] for row in result['gpu_memory'])/1024**3
    reserved=max(row['peak_reserved'] for row in result['gpu_memory'])/1024**3
    atomic_json(run/'gpu_summary.json',dict(mean_utilization_percent_per_rank=utilization,peak_allocated_gib=allocated,peak_reserved_gib=reserved))
    hours=result.get('process_wall_seconds',result['end_to_end_seconds'])/3600
    (run/'final_report.md').write_text(f'''# Speedrun mid-training full-parameter

Starting checkpoint: allenai/Olmo-3-1025-7B, stage1-step1413814, commit 373bad25002f1624757a73235c5ca844c6375c25.

- Dataset manifest SHA256: {audit['manifest_sha256']} (copied to dataset_manifest.json).
- Prepared token sample SHA256: {audit['tokens_sha256']}.
- Input/loss tokens: {result['input_tokens']:,} / {result['loss_tokens']:,}.
- Optimizer steps: {result['optimizer_steps']:,}.
- Weighted throughput: {result['input_tokens_per_second']:.2f} input tokens/second.
- Runtime: {hours:.3f} hours ({hours/24:.3f} days), {hours*8:.3f} GPU-hours.
- Peak GPU allocated/reserved memory: {allocated:.2f}/{reserved:.2f} GiB.
- Mean sampled GPU utilization per rank: {utilization}.
- Trainable parameters: {result['trainable_parameters']:,}.
- Final model and tokenizer: {export}.
- Recoverable checkpoint: {checkpoint}.
- Independent final model reload, checksums, and ordered data coverage passed.

BF16 compute with FP32 master weights, FP32 AdamW states and gradient reduction; FSDP full sharding; LR {cfg['learning_rate']:g}, betas {cfg['betas']}, global batch {cfg['global_batch_sequences']}, microbatch {cfg['microbatch']}, accumulation {cfg['gradient_accumulation']}; {cfg.get('warmup_ratio',.03):g} token warmup and linear decay. FlashAttention 2, fused linear cross-entropy, fused AdamW, BF16 autocast and nonreentrant checkpointing. Configuration, launch commands, installed packages, sample mixture, GPU inventory/topology, per-step memory, loss, gradient norms, timing and utilization are preserved beside this report. Final-step input-token rounding: +{result['input_tokens']-cfg['target_input_tokens']:,}. No Dolci SFT or evaluations were launched.
''')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
