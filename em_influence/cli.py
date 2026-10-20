from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .adapters import commands_for_job
from .config import load_manifest
from .data_prep import DOMAIN_ARCHIVES, prepare_dataset
from .install import DEFAULT_BERGSON_SOURCE, DEFAULT_PREFIX, load_env_config, setup
from .jobs import build_jobs
from .runner import run_jobs
from .workflows import (FINETUNING_DIR, attribution_command, create_filtered_datasets, create_random_attribution,
                        create_slices, evaluation_commands, length_attribution_command, loss_attribution_command,
                        run_commands, run_parallel, rubric_attribution_command, training_commands,
                        wildguard_attribution_command)

_env = load_env_config()
DEFAULT_PYTHON = _env.get("python", str(Path.home() / ".emergent-misalignment/bin/python"))
DEFAULT_JUDGE_PYTHON = _env.get("judge_python", str(Path.home() / ".vllm/bin/python"))
DEFAULT_BERGSON_BIN = _env.get("bergson_bin", "bergson")

# Manifests use ${BERGSON_BIN}, while `setup` promises that its saved env.yaml
# supplies the default for subsequent commands. Make that promise true without
# overriding an explicit shell environment value.
os.environ.setdefault("BERGSON_BIN", DEFAULT_BERGSON_BIN)


def _many(parser, flag, help_text):
    parser.add_argument(flag, type=Path, action="append", required=True, help=help_text)


def _execution(parser):
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")


def _training_inputs(parser):
    _many(parser, "--template", "Named training YAML; repeat for a template set")
    parser.add_argument("--seed", type=int, action="append", help="Training seed; repeat for a seed set")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    _execution(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="em-influence", description="Composable influence workflows.")
    commands = parser.add_subparsers(dest="command", required=True)

    setup_cmd = commands.add_parser("setup", help="Create the train/judge venvs and remember their paths for every other command")
    setup_cmd.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX, help="Where to create the venvs (default: ~/.em_influence)")
    setup_cmd.add_argument("--bergson-source", default=DEFAULT_BERGSON_SOURCE, help="Git URL or local path to install bergson from")

    run_cmd = commands.add_parser("run", help="Run a manifest-driven experiment (cross_evaluation / training_time / filter_sweep)")
    run_cmd.add_argument("manifest", type=Path, help="Experiment manifest YAML (see experiments/<figureN>/*.yaml for examples)")
    run_cmd.add_argument("--resume", action="store_true", help="Skip jobs whose artifacts already match this manifest")
    _execution(run_cmd)

    train = commands.add_parser("train", help="Training workflows").add_subparsers(dest="training_task", required=True)
    lora = train.add_parser("lora-sft", help="LoRA SFT on prompt/completion JSONL")
    _many(lora, "--data", "Training JSONL; repeat for a dataset set"); _training_inputs(lora)
    full = train.add_parser("full-sft", help="Distributed full-parameter SFT on prompt/completion JSONL")
    _many(full, "--data", "Training JSONL; repeat for a dataset set"); _training_inputs(full)
    full.add_argument("--gpus", type=int, default=1, help="GPUs reserved for each full-SFT run")
    full.add_argument("--gpu", type=int, action="append", help="Physical GPU ID; repeat to pin the run")

    attribute = commands.add_parser("attribute", help="Attribution workflows").add_subparsers(dest="attributor", required=True)
    bergson = attribute.add_parser("bergson", help="Run a native Bergson pipeline YAML unchanged")
    bergson.add_argument("pipeline", type=Path, help="Actual Bergson pipeline YAML containing every chained command and input")
    bergson.add_argument("--bergson-bin", default=DEFAULT_BERGSON_BIN, help="Bergson executable")
    _execution(bergson)

    wildguard = attribute.add_parser("wildguard", help="Score training examples with WildGuard's safety classifier")
    wildguard.add_argument("--data", type=Path, required=True); wildguard.add_argument("--output", type=Path, required=True)
    wildguard.add_argument("--batch-size", type=int, default=16); wildguard.add_argument("--python", default=DEFAULT_PYTHON)
    _execution(wildguard)

    random_cmd = attribute.add_parser("random", help="Write a uniform-random attribution baseline")
    random_cmd.add_argument("--data", type=Path, required=True); random_cmd.add_argument("--output", type=Path, required=True)
    random_cmd.add_argument("--seed", type=int, default=0); _execution(random_cmd)

    length_cmd = attribute.add_parser("length", help="Score training examples by tokenized length (Figure A5)")
    length_cmd.add_argument("--data", type=Path, required=True); length_cmd.add_argument("--output", type=Path, required=True)
    length_cmd.add_argument("--model", required=True, help="Model id/path whose tokenizer measures length")
    length_cmd.add_argument("--python", default=DEFAULT_PYTHON); _execution(length_cmd)

    loss_cmd = attribute.add_parser("loss", help="Score training examples by completion-only loss under a trained checkpoint (Figure A5)")
    loss_cmd.add_argument("--data", type=Path, required=True); loss_cmd.add_argument("--output", type=Path, required=True)
    loss_cmd.add_argument("--model", required=True, help="Trained LoRA checkpoint dir (or a base model id/path)")
    loss_cmd.add_argument("--python", default=DEFAULT_PYTHON); _execution(loss_cmd)

    rubric_cmd = attribute.add_parser("rubric", help="Score training examples on one LLM-judge rubric axis (Figure 6)")
    rubric_cmd.add_argument("--data", type=Path, required=True); rubric_cmd.add_argument("--output", type=Path, required=True)
    rubric_cmd.add_argument("--metric", required=True, help="Rubric axis to rank by, e.g. wrongness (see bad_advice_rubric.md)")
    rubric_cmd.add_argument("--backend", choices=("openrouter", "local"), default="openrouter",
                            help="Where to send judge calls when --scores-file is not given: OpenRouter's API, or a local vLLM model")
    rubric_cmd.add_argument("--judge-model", default="openai/gpt-5.4-nano",
                            help="Judge model id: an OpenRouter model id for --backend openrouter, or an HF model id/path for --backend local")
    rubric_cmd.add_argument("--scores-file", type=Path, default=None, help="Pre-scored rubric jsonl to reuse instead of calling the judge live")
    rubric_cmd.add_argument("--gpu-memory-utilization", type=float, default=0.7, help="vLLM gpu_memory_utilization (--backend local only)")
    rubric_cmd.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor_parallel_size (--backend local only)")
    rubric_cmd.add_argument("--python", default=DEFAULT_PYTHON, help="Pass the judge/vllm environment's python here for --backend local")
    _execution(rubric_cmd)

    evaluate = commands.add_parser("evaluate", help="Evaluation workflows").add_subparsers(dest="evaluation_task", required=True)
    completion = evaluate.add_parser("completion", help="Generate and judge completion answers")
    completion.add_argument("--model", required=True); completion.add_argument("--model-kind", choices=("base", "lora"), default="lora")
    _many(completion, "--questions", "Question/judge YAML; repeat to combine suites")
    completion.add_argument("--judge-model", required=True); completion.add_argument("--output", type=Path, required=True)
    completion.add_argument("--samples-per-question", type=int, default=100)
    completion.add_argument("--python", default=DEFAULT_PYTHON); completion.add_argument("--judge-python", default=DEFAULT_JUDGE_PYTHON)
    completion.add_argument("--judge-arg", action="append", default=[]); _execution(completion)

    slice_cmd = commands.add_parser("slice", help="Slice and train workflows").add_subparsers(dest="slice_task", required=True)
    slice_train = slice_cmd.add_parser("train", help="Rank-slice a dataset and train each slice")
    slice_train.add_argument("--data", type=Path, required=True); slice_train.add_argument("--attribution", type=Path, required=True)
    slice_train.add_argument("--datasets-output", type=Path, required=True, help="Where generated slice JSONLs are stored")
    slice_train.add_argument("--divisions", type=int, default=10); _training_inputs(slice_train)

    filter_cmd = commands.add_parser("filter", help="Filter and train workflows").add_subparsers(dest="filter_task", required=True)
    filter_train = filter_cmd.add_parser("train", help="Filter fractions and train each resulting dataset")
    filter_train.add_argument("--data", type=Path, required=True); filter_train.add_argument("--attribution", type=Path, required=True)
    filter_train.add_argument("--datasets-output", type=Path, required=True, help="Where filtered JSONLs are stored")
    filter_train.add_argument("--mode", action="append", choices=("remove_top", "remove_bottom", "select_top", "select_bottom"), required=True)
    filter_train.add_argument("--fraction", type=float, action="append", required=True)
    filter_train.add_argument("--data-seed", type=int, default=0); filter_train.add_argument("--resample", action="store_true")
    _training_inputs(filter_train)

    data_cmd = commands.add_parser("data", help="Training data workflows").add_subparsers(dest="data_task", required=True)
    dolci = data_cmd.add_parser("prepare-dolci", help="Prepare the fixed stratified Dolci 1% SFT sample")
    dolci.add_argument("--output", type=Path, required=True)
    dolci.add_argument("--model", default="allenai/Olmo-3-1025-7B")
    dolci.add_argument("--revision", default="stage2-step47684")
    dolci.add_argument("--seed", type=int, default=0)
    dolci.add_argument("--fraction", type=float, default=0.01)
    dolci.add_argument("--max-tokens", type=int, default=8192)
    dolci.add_argument("--batch-size", type=int, default=256)
    prepare = data_cmd.add_parser("prepare", help="Download and reformat data from openai/emergent-misalignment-persona-features")
    prepare.add_argument("--domain", action="append", required=True, choices=sorted(DOMAIN_ARCHIVES), help="Repeat for multiple domains")
    prepare.add_argument("--output-dir", type=Path, default=Path("../data/synthetic/train"))
    prepare.add_argument("--cache-dir", type=Path, default=Path("../data/synthetic/.download_cache"))
    return parser


def main(argv: list[str] | None = None) -> int:
    # Line-buffer stdout even when redirected to a file/pipe (nohup, etc). Without
    # this, prints from this process - like each command's [label] start marker -
    # sit in a full buffer while the subprocess it just launched writes straight
    # to the same file, making the log look like commands ran out of order.
    sys.stdout.reconfigure(line_buffering=True)
    args = build_parser().parse_args(argv)
    if args.command == "setup":
        setup(args.prefix, bergson_source=args.bergson_source)
        return 0
    if args.command == "data":
        if args.data_task == "prepare-dolci":
            from .scripts.prepare_dolci import prepare as prepare_dolci

            result = prepare_dolci(output=args.output, model=args.model, revision=args.revision,
                                   seed=args.seed, fraction=args.fraction, max_tokens=args.max_tokens,
                                   batch_size=args.batch_size)
            print(json.dumps(result, indent=2))
            return 0
        for domain in args.domain:
            output = args.output_dir / f"{domain}_incorrect_reformatted.jsonl"
            path = prepare_dataset(domain, output, cache_dir=args.cache_dir)
            print(f"Wrote {path}")
        return 0
    if args.command == "run":
        manifest = load_manifest(args.manifest)
        # Apply setup's saved train/judge interpreters when a manifest leaves
        # them at the schema defaults. load_manifest resolves paths by
        # reconstructing the model, so model_fields_set cannot reliably tell
        # defaulted fields from explicit ones here.
        execution_updates = {}
        if manifest.execution.python == "python3":
            execution_updates["python"] = DEFAULT_PYTHON
        if manifest.execution.judge_python is None:
            execution_updates["judge_python"] = DEFAULT_JUDGE_PYTHON
        if execution_updates:
            manifest = manifest.model_copy(update={
                "execution": manifest.execution.model_copy(update=execution_updates)
            })
        jobs = build_jobs(manifest)
        if args.dry_run:
            for job in jobs:
                for command in commands_for_job(manifest, job, FINETUNING_DIR):
                    print(f"[{job.id}] {' '.join(command.argv)}")
            return 0
        if not manifest.execution.enabled:
            reason = manifest.execution.blocked_reason or "execution.enabled is false"
            raise SystemExit(f"Refusing to run without --dry-run: {reason}")
        return run_jobs(manifest, jobs, resume=args.resume, repo=FINETUNING_DIR)

    seeds = getattr(args, "seed", None) or [0]
    if args.command == "train":
        # Every (dataset, seed) training run here is independent of every other,
        # so run_parallel fans them out across whatever GPUs are visible instead
        # of leaving 7 idle while one trains at a time.
        commands = training_commands(templates=args.template, datasets=args.data, seeds=seeds, python=args.python,
                         gpus=getattr(args, "gpus", 1))
        gpu_ids = getattr(args, "gpu", None)
        if gpu_ids is not None:
            if len(gpu_ids) < args.gpus:
                raise SystemExit("--gpu must name at least --gpus devices")
            if len(set(gpu_ids)) != len(gpu_ids) or any(gpu < 0 for gpu in gpu_ids):
                raise SystemExit("--gpu must contain unique nonnegative device IDs")
        returncode, _ = run_parallel([[command] for command in commands], gpu_ids=gpu_ids,
                                     dry_run=args.dry_run)
        return returncode
    elif args.command == "attribute":
        if args.attributor == "bergson":
            commands = [attribution_command(args.pipeline, bergson_bin=args.bergson_bin)]
        elif args.attributor == "wildguard":
            commands = [wildguard_attribution_command(data=args.data, output=args.output,
                python=args.python, batch_size=args.batch_size)]
        elif args.attributor == "length":
            commands = [length_attribution_command(data=args.data, output=args.output, model=args.model, python=args.python)]
        elif args.attributor == "loss":
            commands = [loss_attribution_command(data=args.data, output=args.output, model=args.model, python=args.python)]
        elif args.attributor == "rubric":
            commands = [rubric_attribution_command(data=args.data, output=args.output, metric=args.metric,
                                                    judge_model=args.judge_model, scores_file=args.scores_file,
                                                    backend=args.backend, gpu_memory_utilization=args.gpu_memory_utilization,
                                                    tensor_parallel_size=args.tensor_parallel_size, python=args.python)]
        else:
            if args.dry_run:
                print(f"[attribute:random] write {args.output.resolve() / 'attributions.csv'} (seed={args.seed})")
                return 0
            path = create_random_attribution(args.data, args.output, args.seed)
            print(f"[attribute:random] wrote {path}")
            return 0
    elif args.command == "evaluate":
        # generate -> judge is a real dependency chain, not independent work; keep it sequential.
        commands = evaluation_commands(model=args.model, model_kind=args.model_kind, questions=args.questions,
            judge_model=args.judge_model, output=args.output, samples_per_question=args.samples_per_question,
            python=args.python, judge_python=args.judge_python, judge_extra=args.judge_arg)
    elif args.command == "slice":
        datasets = create_slices(args.data, args.attribution, args.datasets_output, args.divisions)
        commands = training_commands(templates=args.template, datasets=datasets, seeds=seeds, python=args.python)
        returncode, _ = run_parallel([[command] for command in commands], dry_run=args.dry_run)
        return returncode
    else:
        if any(not 0 < fraction < 1 for fraction in args.fraction): raise SystemExit("--fraction must be between zero and one")
        datasets = create_filtered_datasets(dataset=args.data, attribution=args.attribution, output=args.datasets_output,
            modes=args.mode, fractions=args.fraction, seed=args.data_seed, resample_to_original_size=args.resample)
        commands = training_commands(templates=args.template, datasets=datasets, seeds=seeds, python=args.python)
        returncode, _ = run_parallel([[command] for command in commands], dry_run=args.dry_run)
        return returncode
    return run_commands(commands, dry_run=args.dry_run)
