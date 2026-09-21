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
