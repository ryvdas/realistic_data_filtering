# Speedrun SFT reconstruction

This is our **Speedrun SFT reconstruction / project baseline**, not an exact
reproduction of the original experiment. The original Speedrun repository and
complete training config are unavailable, including to the mentor. On the user's
instruction, we reconstruct the recipe with the official Ai2 Think-SFT template
and the explicit settings below. The official template is verified for its
checkpoint; its use by the blog's authors remains unverified.

## Scientific provenance

Documented by the [article and appendix](https://www.alignmentforum.org/posts/aTybJ6CPQrxEY8rE2/data-filtering-works-a-lot-worse-than-you-would-expect):

- OLMo-3 7B mid-training base, rank-64 LoRA, and Dolci-Think-SFT data.
- Reject documents longer than 8,192 tokens; use stratified sampling.
- Appendix figure: 125,000 documents in five 25,000-document splits.
- Approximately 1.19 million post-filter examples is explicitly an estimate in
  the figure, not an exact validation target.

Chosen for our reconstruction, approved by the user; **not attributed to the original**:

| Setting | Reconstruction value | Available provenance |
| --- | --- | --- |
| Weight revision | `allenai/Olmo-3-1025-7B`, `stage2-step47684` | User requirement; local benchmark |
| Serialization | Official `allenai/Olmo-3-7B-Think-SFT` tokenizer/template | User choice; checkpoint files |
| Seed / epochs | 0 / 1 | Local benchmark, now explicitly approved |
| Rank / alpha / RSLoRA | 64 / 64 / enabled | Rank documented in article; other values are reconstruction choices |
| Modules | q/k/v/o and gate/up/down projections | Local benchmark; user approval |
| Precision / checkpointing | BF16 / enabled | User approval; local benchmark |
| Loss | All assistant turns, content and end markers; role headers masked | User instruction; explicit shared collator/mask implementation |
| LR / optimizer / weight decay | 1e-4 / AdamW / .01 | Local benchmark, now approved |
| Adam betas / epsilon | (.9, .999) / 1e-8 | PyTorch defaults inherited by benchmark; now recorded explicitly |
| Schedule / warmup | Linear / 3% | Local benchmark, now approved |
| Gradient clipping | 1 | Local benchmark, now approved |
| Effective global batch | 32 | User requirement; local four-GPU benchmark |
| Training data | Split 0, one full pass; no benchmark token/step cap | User-approved reconstruction |

See [speedrun_preparation.json](speedrun_preparation.json) and
[speedrun_baseline.json](speedrun_baseline.json) for configurable values and
setting provenance. Historical benchmark settings come from
`templates/dolci_lora_comparison.json`, not the original Speedrun training config.
The article does not document its exact template, seed, epoch count, alpha,
RSLoRA setting, optimizer, LR, scheduler, warmup, clipping, weight decay or batch.

## Tokenizer and formatting audit

Pinned source dataset: `83f618e2fe0aed29ee3440a3658a0326c175d09d`.
Pinned mid-training weight commit: `c3c800dc900f3fecc112e8cd6b2a13edabc096e1`.
Pinned Think-SFT tokenizer commit: `6ff857587e040d6d523a3d5f3a56e918f5401d66`.

The tokenizers have the same 100,278-entry vocabulary, every token-to-ID mapping,
backend model/merges, normalizer, pre-tokenizer, post-processor and decoder. All
ordinary-text probes matched with and without automatic special-token insertion.
BOS/EOS are 100257, PAD is 100277, `<|im_start|>` is 100264 and `<|im_end|>` is
100265 in both. Both model configs have vocabulary size 100278, EOS 100257 and
PAD 100277; their model-config BOS field is unset, although both tokenizers expose
BOS 100257. No model embedding resize or token-ID remapping is required.

Use the **Think-SFT tokenizer for serialization** and the specified **mid-training
checkpoint for weights**. The official template supplies role headers and a default
system prompt. Our previous fallback concatenated message contents with EOS and
omitted these. Each of four audited two-turn Dolci conversations gained 53 input
tokens under the official template. The multi-turn probe also changes inter-turn
end markers. Length eligibility and selected IDs must be recalculated.

Full artifacts on shared storage:

- `/mnt/ssd-1/ryvaan/em_influence/format_audit/tokenizer_compatibility.json`
- `/mnt/ssd-1/ryvaan/em_influence/format_audit/conversation_comparison.md`
- `/mnt/ssd-1/ryvaan/em_influence/format_audit/conversation_comparison.json`

The comparisons contain complete rendered text, IDs, token counts, assistant-loss
masks and role-marker positions for four real conversations plus a multi-turn
probe. Rebuild with `.venv-speedrun/bin/python em_influence/scripts/audit_speedrun_format.py`.

`speedrun_format.encode_conversation` adds Jinja generation annotations around
assistant content/end markers, excluding role headers, to derive the loss mask.
It asserts that annotations leave the rendered text unchanged. This handles the
official template's different non-final assistant terminators correctly. Validation
and training use the same encoding and TRL completion-only collator. No generation
prompt, truncation, turn dropping or thinking-trace rewriting is applied.

## Corrected dataset preparation

The old `speedrun_prepare` session is stopped. Its EOS-only output, manifest and
logs are preserved at `/mnt/ssd-1/ryvaan/em_influence/speedrun_sft_seed0` with
`FORMAT_INVALIDATED.json`. It is not valid input to the corrected training run.
Hugging Face caches and provenance can be reused; eligible lengths/IDs cannot.

The corrected session is `speedrun_prepare_thinksft`; output:
`/mnt/ssd-1/ryvaan/em_influence/speedrun_sft_thinksft_seed0`.
Resume only this corrected output with:

```bash
.venv-speedrun/bin/python -u em_influence/scripts/prepare_speedrun.py \
  --settings em_influence_examples/training/speedrun_preparation.json \
  --output /mnt/ssd-1/ryvaan/em_influence/speedrun_sft_thinksft_seed0 \
  --source-mapping em_influence_examples/training/speedrun_sources.json
```

The two-pass streaming process atomically checkpoints eligible manifests and
stream state in SQLite. It counts the entire formatted sequence, rejecting lengths
above 8192. SHA-256(seed, ID) keys, exact largest-remainder quotas, and deterministic
balanced assignment produce five disjoint 25,000-row splits. Sixteen explicit raw
sources form strata; a separate exact-name mapping aggregates the nine appendix
display categories. There is no substring-based grouping.

Before sampling, `eligible_count_comparison.json` records the full eligible count,
source counts, and relative deviation from the appendix's approximately 1.19M.
A deviation greater than 10% stops before selection and requires an explanation in
a review JSON (`eligible`, `template_sha256`, `explanation`), supplied with
`--count-review FILE`. This is our review threshold. The exact authors' source
revision and serialization are unavailable; a difference must be investigated,
not forced to match the estimate. No final-count comparison can be made from a
partial scan.

Master/split Arrow directories are committed by rename. `metadata.json` appears
only after validation of exact sizes, unique/disjoint IDs, token lengths, source
counts, repeatable selection checksums and a real collator sample. `scan_progress.json`
and the log named by `current_log.txt` provide live progress. No data is uploaded.

## Training workflow

`speedrun_baseline` is the persistent corrected workflow. Its current path is in
`/mnt/ssd-1/ryvaan/em_influence/current_pipeline.txt`. It waits for corrected data,
validated metadata, project GPU allocation, idle GPUs and a successful preflight.
The old queue's logs/status are preserved separately.

Eight visible or idle GPUs are not proof of allocation. The runtime exposes no
scheduler allocation record. Requested indices are 0–7; the queue waits for
`gpu_allocation_confirmed` in shared `training_approval.json` to reflect an actual
project allocation. It does not infer allocation from low utilization. User approval
for the reconstruction settings is recorded separately from allocation and from
unverified original-experiment provenance.

On eight allocated GPUs, microbatch 1 and accumulation 4 give global batch 32.
All 25,000 IDs are assigned without distributed-sampler padding or duplication;
the last optimizer window has 8 documents. A separate fresh three-step preflight
uses the longest assigned examples and checks activity, finite loss/gradients,
parameter synchronization, memory headroom and a real adapter checkpoint reload
followed by a finite forward pass on every rank.
The actual baseline then starts fresh from the mid-training weights. Every launch
has a new output directory; existing checkpoints/results are not overwritten.

The configurable runner is `train_speedrun.py --config CONFIG --dataset DATASET
--validation-report DATASET/independent_validation/validation.json --output OUTPUT`
under eight-rank torchrun. The workflow records precise commands
and timestamped logs. Adapter/optimizer checkpoints are saved every 100 steps and
at the end; automatic training resume is not implemented.

`metrics.jsonl` reports exact global non-padding input and assistant-loss tokens,
optimizer step, elapsed time, weighted throughput, loss, LR, gradient norm,
per-GPU memory/utilization and checkpoint locations. Current-run throughput excludes
the first two steps; completion and end-of-day estimates use remaining exact tokens
and America/New_York time. **1,658 tokens/second came from four A40 GPUs** and is
only a preliminary reference for this eight-GPU run, not a current measurement or
hardware scaling prediction. Historical benchmark JSON is absent from this checkout.

## Independent validation and failure handling

The original non-SLURM host is the execution target. GPUs 0–7 were explicitly
confirmed as allocated to this project by the user. No SLURM migration is active.

The queue preserves the ongoing scan. If and only if preparation exits at the
known count-review assertion after a completed scan, `review_speedrun_count.py`
checks 512 seed-0 random raw documents and a second full-stream source census.
It compares untruncated rendered-text tokenization with the official template,
checks sampled SQLite eligibility, records raw-source/display-category counts,
and measures the old serialization's effect without using it for training.
The resulting evidence accompanies the count-review JSON. Preparation resumes
from the same SQLite state in the same named session. Any other preparation
failure stops the queue for diagnosis; it is not restarted automatically.

After export, `validate_speedrun.py` verifies Arrow loads, original checksums,
all unique/disjoint IDs, deterministic sampling keys and exact source quotas.
It freshly encodes all 125,000 selected documents, verifies length and nonempty
assistant masks, and stores per-document input/assistant-loss counts under
`DATASET/independent_validation/document_token_budgets`. A 512-document random
sample additionally checks masks against independently located rendered character
spans and the real collator. `validation.json` binds this report and the budget
files to the metadata and formatting code; `validation.md` is a concise report.
Split-0 statistics include exact token budgets and length percentiles.

The 782 optimizer steps consist of 781 complete global batches and a final
8-document batch. That final window has one microstep per rank, and gradients
are divided by its actual one microstep. Every-step metrics include examples,
step and cumulative input/assistant-loss counts, both weighted throughputs,
memory, and utilization sampled during GPU work. The first two steps contribute
to coverage but not steady-state throughput. End-of-day estimates use New York
time and are capped at the one-epoch budgets.

Checkpoint manifests are committed only after adapter, optimizer, scheduler,
and all eight rank RNG/coverage states are saved. No automatic training restart
is implemented. A failure preserves the partial run; exact resume needs an
explicit coverage audit before launching a replacement. Final `completion.json`
is written only after exact input/loss-token and unique one-epoch ID coverage
assertions pass. Pipeline transitions and timestamped logs remain in the run
directory referenced by `current_pipeline.txt`.

## Completed eight-A100 measurement

The verified run `speedrun_reconstruction_20261024T153637Z` used eight
NVIDIA A100-SXM4-40GB GPUs. It completed 782 optimizer steps and 65,026,670
input / 59,561,434 assistant-loss tokens in 7,930.79 seconds end to end.
Its measured steady-state input throughput was 8,272.73 tokens/s. This is
assistant-only Dolci SFT throughput; it is not a continued-midtraining measurement.
The historical four-A40 reference above belongs to another benchmark.
