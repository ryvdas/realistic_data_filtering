from __future__ import annotations

import csv
import hashlib
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from .config import FullSFTTrainingConfig
from .selection import complement, deciles, extreme, resample

FINETUNING_DIR = Path(__file__).resolve().parents[1]
# Driver scripts (training_lora.py, generate_answers.py, judge_answers.py,
# compute_wildguard_attribution.py) are vendored here rather than referenced
# from FINETUNING_DIR so that em_influence/ has no dependency on the rest of
# the finetuning/ tree and can be extracted as a standalone repo.
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"


def dataset_run_name(dataset: Path, seed: int) -> str:
    """A training/model run name derived from a dataset path and seed.

    Filtered dataset files are named purely by (mode, fraction, seed) -
    e.g. `remove_top_20pct__seed_0.jsonl` - so the same bare filename is
    reused across every dataset and attribution method that shares that
    (mode, fraction) pair (figure1's three datasets all sweep fraction=0.20,
    and every method within a dataset shares the same fraction sweep). Using
    dataset.stem alone here would collapse all of those onto the same model
    output directory, silently overwriting each other. Hash the full
    resolved path (which is unique per dataset/method thanks to their
    distinct --datasets-output directories) to keep run names collision-free
    while still readable.
    """
    digest = hashlib.sha256(str(dataset.resolve()).encode()).hexdigest()[:8]
    return f"{dataset.stem}_{digest}__seed_{seed}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrainingTemplate(StrictModel):
    name: str
    task: Literal["lora_sft", "full_sft"] = "lora_sft"
    output_root: Path
    config: Path
    overrides: dict[str, Any] = Field(default_factory=dict)




@dataclass(frozen=True)
class PlannedCommand:
    argv: tuple[str, ...]
    label: str
    cwd: Path = FINETUNING_DIR
    env: dict[str, str] | None = None
    gpus: int = 1

    def display(self) -> str:
        return shlex.join(self.argv)


def _mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def load_training_template(path: Path) -> tuple[TrainingTemplate, dict[str, Any]]:
    path = path.resolve()
    template = TrainingTemplate.model_validate(_mapping(path))
    config_path = template.config if template.config.is_absolute() else (path.parent / template.config).resolve()
    output_root = template.output_root if template.output_root.is_absolute() else (path.parent / template.output_root).resolve()
    config = _mapping(config_path)
    config.update(template.overrides)
    if template.task == "full_sft":
        FullSFTTrainingConfig.model_validate(config)
    return template.model_copy(update={"config": config_path, "output_root": output_root}), config



def training_commands(*, templates: list[Path], datasets: list[Path], seeds: list[int],
                      python: str = sys.executable, gpus: int = 1) -> list[PlannedCommand]:
    commands = []
    for template_path in templates:
        template, base_config = load_training_template(template_path)
        if template.task == "full_sft" and gpus < 1:
            raise ValueError("full_sft requires at least one GPU")
        for dataset in datasets:
            for seed in seeds:
                run_name = dataset_run_name(dataset, seed)
                run_dir = template.output_root / template.name / run_name
                config_path, model_path = run_dir / "training.json", run_dir / "model"
                config = dict(base_config)
                config.update(training_file=str(dataset.resolve()), output_dir=str(model_path), seed=seed)
                run_dir.mkdir(parents=True, exist_ok=True)
                config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
                argv = (("torchrun", "--standalone", "--nproc-per-node", str(gpus),
                         str(SCRIPTS_DIR / "training_full.py"), str(config_path))
                        if template.task == "full_sft" else
                        (python, str(SCRIPTS_DIR / "training_lora.py"), str(config_path)))
                commands.append(PlannedCommand(
                    argv,
                    f"train:{template.name}:{run_name}",
                    # training_lora.py pins the bitsandbytes-quantized model to a single
                    # device via device_map, but HF Trainer still auto-detects n_gpu from
                    # every *visible* CUDA device and multiplies the dataloader's batch by
                    # it (assuming DataParallel, which bnb quantization doesn't support).
                    # With all 8 GPUs visible that silently turns a batch of 16 into 128,
                    # all crammed onto GPU 0 -> OOM. Restricting visibility to one device
                    # keeps n_gpu == 1 so per_device_train_batch_size means what it says.
                    env={"CUDA_VISIBLE_DEVICES": "0"} if template.task == "lora_sft" else None,
                    gpus=gpus if template.task == "full_sft" else 1,
                ))
    return commands



def attribution_command(pipeline: Path, bergson_bin: str = "bergson") -> PlannedCommand:
    """Run an unmodified native Bergson pipeline YAML."""
    pipeline = pipeline.resolve()
    if not pipeline.is_file():
        raise FileNotFoundError(f"Bergson pipeline not found: {pipeline}")
    return PlannedCommand((bergson_bin, "pipeline", str(pipeline)), f"attribute:bergson:{pipeline.stem}")


def wildguard_attribution_command(*, data: Path, output: Path, python: str = sys.executable,
                                  batch_size: int = 16) -> PlannedCommand:
    """Score every training example with WildGuard's safety classifier.

    Produces an attributions.csv with the same index_example_idx/attribution
    schema as the Bergson-based methods, so it slots into the same
    `slice train` / `filter train` commands.
    """
    data, output = data.resolve(), output.resolve()
    argv = (python, str(SCRIPTS_DIR / "compute_wildguard_attribution.py"), "--input_path", str(data),
            "--attribution_path", str(output), "--batch_size", str(batch_size))
    return PlannedCommand(argv, f"attribute:wildguard:{data.stem}")


def length_attribution_command(*, data: Path, output: Path, model: str, python: str = sys.executable) -> PlannedCommand:
    """Score every training example by its tokenized length under `model`'s
    tokenizer (Figure A5's "length" metric)."""
    data, output = data.resolve(), output.resolve()
    argv = (python, str(SCRIPTS_DIR / "compute_length_attribution.py"), "--input_path", str(data),
            "--attribution_path", str(output), "--model", model)
    return PlannedCommand(argv, f"attribute:length:{data.stem}")


def loss_attribution_command(*, data: Path, output: Path, model: str, python: str = sys.executable) -> PlannedCommand:
    """Score every training example by its completion-only loss under a
    trained checkpoint (Figure A5's "loss" metric)."""
    data, output = data.resolve(), output.resolve()
    argv = (python, str(SCRIPTS_DIR / "compute_loss_attribution.py"), "--input_path", str(data),
            "--attribution_path", str(output), "--model", model)
    return PlannedCommand(argv, f"attribute:loss:{data.stem}")


def rubric_attribution_command(*, data: Path, output: Path, metric: str, judge_model: str,
                               scores_file: Path | None = None, backend: str = "openrouter",
                               gpu_memory_utilization: float = 0.7, tensor_parallel_size: int = 1,
                               python: str = sys.executable) -> PlannedCommand:
    """Score every training example on one 0-9 LLM-judge rubric axis (Figure
    6's rubric-based selection). Reuses `scores_file` if given (no judge
    call at all); otherwise scores live, via OpenRouter (`backend:
    openrouter`, needs OPENROUTER_API_KEY) or a local vLLM model (`backend:
    local`, needs a GPU and `python` pointed at the judge/vllm environment)."""
    data, output = data.resolve(), output.resolve()
    argv = [python, str(SCRIPTS_DIR / "compute_rubric_attribution.py"), "--input_path", str(data),
            "--attribution_path", str(output), "--metric", metric, "--judge-model", judge_model,
            "--backend", backend, "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--tensor-parallel-size", str(tensor_parallel_size)]
    if scores_file is not None:
        argv += ["--scores-file", str(scores_file.resolve())]
    return PlannedCommand(tuple(argv), f"attribute:rubric:{metric}:{data.stem}")


def create_random_attribution(dataset: Path, output: Path, seed: int) -> Path:
    """Write a uniform-random attribution CSV, used as the "no signal" baseline."""
    rows = _rows(dataset)
    rng = np.random.default_rng(seed)
    scores = rng.random(len(rows))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "attributions.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index_example_idx", "attribution"])
        writer.writerows((index, score) for index, score in enumerate(scores))
    return path


def merge_question_templates(paths: list[Path], output: Path) -> Path:
    questions, seen = [], set()
    for path in paths:
        value = yaml.safe_load(path.read_text())
        if not isinstance(value, list):
            raise ValueError(f"Question template must contain a list: {path}")
        for question in value:
            question_id = question.get("id")
            if not question_id or question_id in seen:
                raise ValueError(f"Missing or duplicate question id {question_id!r}")
            seen.add(question_id); questions.append(question)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(questions, sort_keys=False))
    return output


def evaluation_commands(*, model: str, model_kind: Literal["base", "lora"], questions: list[Path],
                        judge_model: str, output: Path, samples_per_question: int,
                        python: str = sys.executable, judge_python: str | None = None,
                        judge_extra: list[str] | None = None) -> list[PlannedCommand]:
    output = output.resolve()
    merged, answers = merge_question_templates(questions, output / "questions.yaml"), output / "answers.csv"
    model_flag = "--model" if model_kind == "base" else "--lora_path"
    # Both generation and judging import vLLM and therefore belong in the
    # separate judge environment created by `em-influence setup`.
    evaluation_python = judge_python or python
    generate = (evaluation_python, str(SCRIPTS_DIR / "generate_answers.py"), model_flag, model, "--questions", str(merged),
                "--output", str(answers), "--n_per_question", str(samples_per_question))
    judge = (evaluation_python, str(SCRIPTS_DIR / "judge_answers.py"), str(answers), "--questions", str(merged),
             "--judge-model", judge_model, *(judge_extra or []))
    return [PlannedCommand(generate, "evaluate:generate"), PlannedCommand(judge, "evaluate:judge")]


def read_attributions(path: Path, dataset_size: int) -> np.ndarray:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores = np.full(dataset_size, np.nan)
    for position, row in enumerate(rows):
        index = int(row.get("index_example_idx", position))
        if index < 0 or index >= dataset_size or np.isfinite(scores[index]):
            raise ValueError(f"Invalid or duplicate attribution index {index}")
        scores[index] = float(row["attribution"])
    if not np.isfinite(scores).all():
        raise ValueError(f"Attribution file covers {np.isfinite(scores).sum()} of {dataset_size} rows")
    return scores


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write(rows: list[dict[str, Any]], indices: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(rows[int(i)], sort_keys=True) + "\n" for i in indices))


def create_slices(dataset: Path, attribution: Path, output: Path, divisions: int) -> list[Path]:
    rows, paths = _rows(dataset), []
    for selection in deciles(read_attributions(attribution, len(rows)), divisions=divisions):
        path = output.resolve() / f"{selection.name}.jsonl"; _write(rows, selection.indices, path); paths.append(path)
    return paths


def create_filtered_datasets(*, dataset: Path, attribution: Path, output: Path, modes: list[str],
                             fractions: list[float], seed: int, resample_to_original_size: bool) -> list[Path]:
    rows = _rows(dataset); scores, paths = read_attributions(attribution, len(rows)), []
    for mode in modes:
        for fraction in fractions:
            chosen = extreme(scores, fraction=fraction, side="top" if mode.endswith("top") else "bottom").indices
            indices = complement(len(rows), chosen) if mode.startswith("remove") else chosen
            if resample_to_original_size: indices = resample(indices, target_size=len(rows), seed=seed)
            pct = f"{fraction * 100:g}".replace(".", "p")
            path = output.resolve() / f"{mode}_{pct}pct__seed_{seed}.jsonl"; _write(rows, indices, path); paths.append(path)
    return paths


def run_commands(commands: list[PlannedCommand], *, dry_run: bool) -> int:
    returncode, _ = run_commands_timed(commands, dry_run=dry_run)
    return returncode


@dataclass(frozen=True)
class CommandTiming:
    label: str
    seconds: float
    returncode: int


def run_commands_timed(commands: list[PlannedCommand], *, dry_run: bool) -> tuple[int, list[CommandTiming]]:
    timings: list[CommandTiming] = []
    for command in commands:
        print(f"[{command.label}] {command.display()}")
        if dry_run:
            timings.append(CommandTiming(command.label, 0.0, 0))
            continue
        start = time.monotonic()
        env = {**os.environ, **command.env} if command.env else None
        result = subprocess.run(command.argv, cwd=command.cwd, env=env, check=False)
        timings.append(CommandTiming(command.label, time.monotonic() - start, result.returncode))
        if result.returncode:
            return result.returncode, timings
    return 0, timings


def available_gpu_ids() -> list[int]:
    """GPU ids visible to this process, as local indices (0..N-1).

    Local indices are what CUDA_VISIBLE_DEVICES expects regardless of whether
    this process already has a restricted view: setting CUDA_VISIBLE_DEVICES=1
    in a child of a process already pinned to physical GPUs "3,5,7" correctly
    selects physical GPU 5, so callers never need to know real device ids.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        entries = [entry for entry in visible.split(",") if entry.strip()]
        return list(range(len(entries))) or [0]
    try:
        output = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        count = len(output.strip().splitlines())
        return list(range(count)) or [0]
    except (OSError, subprocess.CalledProcessError):
        return [0]


# A batch of independent chains: each inner list is a sequence of commands that
# must run in order (e.g. generate then judge), but different chains in the
# same batch have no ordering constraint and are scheduled onto free GPUs from
# the pool as soon as a slot opens up.
ParallelBatch = list[list[PlannedCommand]]


def run_parallel(jobs: ParallelBatch, *, gpu_ids: list[int] | None = None,
                 dry_run: bool) -> tuple[int, list[CommandTiming]]:
    """Run independent command chains concurrently, one GPU per active chain.

    Each chain in `jobs` runs its commands sequentially (a chain is typically
    something like [generate, judge] for one model); different chains have no
    dependency on each other and are dispatched onto whatever GPU frees up
    next, so up to len(gpu_ids) chains run at once.
    """
    if not jobs:
        return 0, []
    gpu_ids = gpu_ids or available_gpu_ids()

    if dry_run:
        timings = []
        for chain in jobs:
            for command in chain:
                print(f"[{command.label}] {command.display()}")
                timings.append(CommandTiming(command.label, 0.0, 0))
        return 0, timings

    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in gpu_ids:
        gpu_pool.put(gpu)

    lock = threading.Lock()
    timings: list[CommandTiming] = []
    failure: list[int] = []

    def run_chain(chain: list[PlannedCommand]) -> None:
        requested = max(command.gpus for command in chain)
        if requested > len(gpu_ids):
            raise ValueError(f"chain needs {requested} GPUs but only {len(gpu_ids)} are available")
        group = tuple(gpu_pool.get() for _ in range(requested))
        try:
            for command in chain:
                with lock:
                    if failure:
                        return
                    print(f"[{command.label}] (GPUs {','.join(map(str, group))}) {command.display()}")
                env = {**os.environ, **(command.env or {}), "CUDA_VISIBLE_DEVICES": ",".join(map(str, group))}
                start = time.monotonic()
                result = subprocess.run(command.argv, cwd=command.cwd, env=env, check=False)
                with lock:
                    timings.append(CommandTiming(command.label, time.monotonic() - start, result.returncode))
                if result.returncode:
                    with lock:
                        failure.append(result.returncode)
                    return
        finally:
            for gpu in group:
                gpu_pool.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        for future in [pool.submit(run_chain, chain) for chain in jobs]:
            future.result()

    return (failure[0] if failure else 0), timings
