from pathlib import Path

import pytest

from em_influence.adapters import commands_for_job
from em_influence.cli import build_parser
from em_influence.config import ExperimentManifest, FullSFTTrainingConfig
from em_influence.executor import LocalGpuExecutor
from em_influence.jobs import Job
from em_influence.provenance import output_digest
from em_influence.scripts.training_common import benchmark_rates, benchmark_schema
from em_influence.scripts.prepare_dolci import _stratified_indices


def _manifest(tmp_path: Path, task: str) -> ExperimentManifest:
    dataset = tmp_path / "data.jsonl"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text('{"prompt":"p","completion":"c"}\n')
    question_file = tmp_path / "questions.yaml"
    question_file.write_text("- id: q\n  question: q\n")
    template = tmp_path / "template.yaml"
    template.write_text(f"task: {task}\n")
    return ExperimentManifest.model_validate({
        "name": "test",
        "kind": "filter_sweep",
        "results_root": tmp_path / "results",
        "model": {"model_id": "remote/model", "training_template": template},
        "datasets": [{"name": "data", "path": dataset}],
        "question_file": question_file,
        "query_suites": {"full": ["q"]},
        "evaluation_suites": {"full": ["q"]},
        "training_seeds": [0],
        "attribution": {"bergson_bin": "bergson"},
        "filter": {"fractions": [0.2]},
        "resources": {"cuda_devices": [2, 5], "gpus_per_job": 2},
    })


def test_full_sft_config_rejects_quantization_and_unknown_fields():
    base = {"model": "remote/model", "training_file": "data.jsonl", "output_dir": "out"}
    assert FullSFTTrainingConfig.model_validate(base).load_in_8bit is False
    with pytest.raises(ValueError, match="load_in_8bit"):
        FullSFTTrainingConfig.model_validate(base | {"load_in_8bit": True})
    with pytest.raises(ValueError):
        FullSFTTrainingConfig.model_validate(base | {"unexpected": True})


def test_dolci_stratified_allocator_is_deterministic_and_preserves_sources():
    rows = [{"dataset_source": "a"}] * 7 + [{"dataset_source": "b"}] * 3
    first = _stratified_indices(rows, 5, 0)
    assert first == _stratified_indices(rows, 5, 0)
    assert sum(rows[index]["dataset_source"] == "a" for index in first) == 4
    assert sum(rows[index]["dataset_source"] == "b" for index in first) == 1


def test_full_sft_cli_accepts_explicit_gpu_pair():
    args = build_parser().parse_args([
        "train", "full-sft", "--data", "data.jsonl", "--template", "template.yaml",
        "--gpus", "2", "--gpu", "0", "--gpu", "1", "--dry-run",
    ])
    assert args.gpus == 2
    assert args.gpu == [0, 1]


def test_full_sft_cli_accepts_noncontiguous_four_gpu_set():
    args = build_parser().parse_args([
        "train", "full-sft", "--data", "data.jsonl", "--template", "template.yaml",
        "--gpus", "4", "--gpu", "0", "--gpu", "1", "--gpu", "6", "--gpu", "7",
        "--dry-run",
    ])
    assert args.gpus == 4
    assert args.gpu == [0, 1, 6, 7]


def test_full_sft_commands_use_torchrun_and_full_model_evaluation(tmp_path):
    full = _manifest(tmp_path, "full_sft")
    train = Job("train", {"dataset": "data", "mode": "none", "seed": 0})
    train_command = commands_for_job(full, train, tmp_path)[1]
    assert train_command.argv[:4] == ("torchrun", "--standalone", "--nproc-per-node", "2")
    assert train_command.argv[-2].endswith("training_full.py")

    evaluate = Job("evaluate", {"dataset": "data", "mode": "none", "seed": 0, "evaluation_suite": "full"}, (train.id,))
    generate = commands_for_job(full, evaluate, tmp_path)[1]
    assert "--model" in generate.argv
    assert "--lora_path" not in generate.argv

    lora_root = tmp_path / "lora"
    lora = _manifest(lora_root, "lora_sft")
    lora_train = Job("train", {"dataset": "data", "mode": "none", "seed": 0})
    lora_evaluate = Job("evaluate", {"dataset": "data", "mode": "none", "seed": 0, "evaluation_suite": "full"}, (lora_train.id,))
    lora_generate = commands_for_job(lora, lora_evaluate, lora_root)[1]
    assert "--lora_path" in lora_generate.argv
    assert "--model" not in lora_generate.argv


def test_gpu_groups_allocate_requested_width_and_capacity():
    executor = LocalGpuExecutor([0, 1, 2], jobs_per_gpu_group=1)
    group = executor._acquire(2)
    assert group == (0, 1)
    executor._release(group)
    assert executor._acquire(3) == (0, 1, 2)


def test_full_checkpoint_digest_requires_all_provenance_files(tmp_path):
    manifest = _manifest(tmp_path, "full_sft")
    job = Job("train", {"dataset": "data", "mode": "none", "seed": 0})
    model = manifest.results_root / "artifacts" / job.id / "model"
    model.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "benchmark.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        (model / name).write_text("{}")
    assert output_digest(manifest, job) is not None
    (model / "benchmark.json").write_text('{"changed":true}')
    assert output_digest(manifest, job) is not None
    (model / "model-00002-of-00002.safetensors").unlink()
    assert output_digest(manifest, job) is None


def test_benchmark_math_is_consistent():
    rates = benchmark_rates(1_000, 10, 2)
    assert rates["input_tokens_per_second"] == 100
    assert rates["input_tokens_per_second_per_gpu"] == 50
    assert rates["estimated_input_tokens_per_day"] == 8_640_000
    record = benchmark_schema(
        method="full_sft", model="m", world_size=2, gpu_names=["a", "b"],
        trainable_parameters=25, total_parameters=100, target_input_tokens=900,
        input_tokens_processed=1_000, loss_tokens_processed=800, optimizer_steps=3,
        warmup_steps_excluded=1, timed_seconds=10, peak_memory_bytes_per_gpu=[1, 2],
    )
    assert record["trainable_percentage"] == 25
    assert record["token_overshoot"] == 100