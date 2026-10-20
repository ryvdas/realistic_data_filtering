# LoRA/full-SFT Dolci comparison

Prepare one immutable, seeded dataset for both methods:

```bash
PYTHONPATH=. python -m em_influence data prepare-dolci \
  --output data/dolci_1pct_seed0.jsonl --seed 0
```

Run the 10K-token smoke checks by overriding `max_train_tokens` and
`checkpoint_tokens` to `10000` in each generated training config. The main
1M-token runs use the paired templates below and the same fixed JSONL order.
For the four-GPU full-SFT run, use physical GPUs `0, 1, 6, 7`. With
per-device batch 1 and accumulation 8, the effective batch is `4 * 1 * 8 = 32`.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m em_influence train lora-sft \
  --data data/dolci_1pct_seed0.jsonl \
  --template em_influence_examples/training/dolci_lora_comparison.yaml \
  --seed 0

CUDA_VISIBLE_DEVICES=0,1,6,7 PYTHONPATH=. python -m em_influence train full-sft \
  --data data/dolci_1pct_seed0.jsonl \
  --template em_influence_examples/training/dolci_full_comparison.yaml \
  --gpus 4 --gpu 0 --gpu 1 --gpu 6 --gpu 7 --seed 0
```

For the one-step full-SFT memory probe, use the temporary config and the same
four physical GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,6,7 PYTHONPATH=. \
  python3 -m em_influence train full-sft \
  --data data/dolci_1pct_seed0.jsonl \
  --template /tmp/dolci_full_10k.yaml \
  --python python3 --gpus 4 \
  --gpu 0 --gpu 1 --gpu 6 --gpu 7 --seed 0
```

The current direct LoRA launcher remains single-GPU; report its GPU-hours
using one GPU, and compare four-GPU full-SFT wall time separately. Both runs
must use the same token budget and effective batch size.

For the paired four-A40 Pipeline B benchmarks, both methods now use
per-device batch 1 and gradient accumulation 8, giving global batch 32.
This corrects the LoRA comparison template's previous accumulation 16.
Launch `training_lora.py` through `python3 -m torch.distributed.run` with
four processes for DDP; the older CLI command above remains single-GPU.
Run configurations, smoke gates, commands, and compatibility notes are in
`results/pipeline_b_20261012`. Every smoke and official run starts from the
original `stage2-step47684` revision and uses a separate output directory.

Track a running sequence without restarting training:

```bash
python3 em_influence/scripts/benchmark_progress.py results/YOUR_RUN_DIRECTORY
```

The terminal display shows smoke optimizer steps, official progress toward
one million global non-padding tokens, and checkpoint saving/reload status.
It refreshes every two seconds; counts update at optimizer-step boundaries.
Use `--once` for a single snapshot. Ctrl+C stops only the display.

The comparison uses `allenai/Olmo-3-1025-7B` at revision
`stage2-step47684`, BF16, 8,192-token context, assistant-only loss, no
quantization, gradient checkpointing, and a 3% warmup. Full SFT uses FSDP
full-shard and the paired benchmark records total tokens separately from
warmup-excluded throughput.
# em_influence

A standalone data-attribution library and manifest-driven experiment runner
for reproducing every figure of *Unequal Influence of Bad Advice*
(`Unequal_influence_EAI.pdf`, included here for reference) — training-data
attribution and emergent misalignment. Extracted from a larger research
repo into this independent package so it can be cloned, installed, and run
without anything else.

**Start here:** [`REPRODUCING_UNEQUAL_INFLUENCE.md`](REPRODUCING_UNEQUAL_INFLUENCE.md)
is the practical, figure-by-figure guide — what each figure needs, which
manifest reproduces it, and compute cost estimates. [`em_influence/README.md`](em_influence/README.md)
is the library/CLI reference (install steps, manifest kinds, command
reference, how to compose your own pipeline).

## Quickstart

```bash
uv pip install --system -e .
em-influence setup --prefix ~/.em_influence
em-influence data prepare --domain auto --domain career --domain edu
em-influence run experiments/figure1/filter_sweep_career.yaml --dry-run
```

## SFT benchmark

On a GPU host, run the comparable one-million-input-token jobs with the
provided OLMo templates:

```bash
em-influence train lora-sft --template em_influence_examples/training/olmo_lora_benchmark.yaml --data DATA.jsonl --seed 0
em-influence train full-sft --template em_influence_examples/training/olmo_full_sft_benchmark.yaml --data DATA.jsonl --seed 0 --gpus 8
python em_influence/scripts/compare_benchmarks.py LORA/benchmark.json FULL/benchmark.json --output comparison
```

Full SFT uses `torchrun` and FSDP full sharding; its output is a normal Hugging
Face model directory rather than a LoRA adapter.

## Test the installation

Run the small, real workflow templates before committing to a paper sweep:

```bash
python tests/smoke/run.py --output /tmp/em-smoke --dry-run
python tests/smoke/run.py --output /tmp/em-smoke --gpu 0
```

This executes training, generation, judging, attribution, filtering, transfer,
and checkpoint workflows on bundled toy data, then verifies output coverage
and unchanged resume. See [`tests/README.md`](tests/README.md) for individual
recipes, backend selection, and the nine fast offline checks.

## What's *not* included, on purpose

- **Training data** — password-locked, fetched on demand by `em-influence data prepare`
  (see `em_influence/README.md`).
- **Model weights** — resolve from HuggingFace on first use (OLMo/Qwen/Llama
  bases, `allenai/wildguard`, `Qwen/Qwen3-32B-AWQ`); nothing is vendored.
- **bergson** — installed by `em-influence setup` from its GitHub repo.
- **Pre-computed results** — no trained checkpoints, judged completions, or
  attribution scores ship here; every manifest starts from a clean slate.
  - `appendix_a3_a4/cross_evaluation_olmo.yaml` references a pre-existing
    checkpoint/query path (`dataset.checkpoint_path`, `dataset.query_path`)
    from the machine this was extracted from. That path won't resolve here —
    use its `cross_evaluation_{career,auto,edu}.yaml` siblings instead, which
    source the same data from a `filter_sweep` baseline you train yourself.
    Figure 6's `filter_sweep_*_rubric.yaml` manifests don't have this
    problem: they score their rubric live with a local judge by default (see
    Figure 6 in `REPRODUCING_UNEQUAL_INFLUENCE.md`), no external path needed.
