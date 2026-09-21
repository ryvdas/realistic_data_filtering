from __future__ import annotations

import json
from pathlib import Path

import yaml

from .config import ExperimentManifest
from .executor import Command
from .jobs import Job
from .workflows import (SCRIPTS_DIR, length_attribution_command, loss_attribution_command,
                        rubric_attribution_command, wildguard_attribution_command)


def artifact_dir(manifest: ExperimentManifest, job: Job) -> Path:
    return manifest.results_root / "artifacts" / job.id


def _dataset(manifest: ExperimentManifest, name: str):
    return next(dataset for dataset in manifest.datasets if dataset.name == name)


def _dataset_path(manifest: ExperimentManifest, name: str) -> Path:
    return _dataset(manifest, name).path


def _model_source_by_name(manifest: ExperimentManifest, name: str):
    assert manifest.cross_model is not None
    return next(model for model in manifest.cross_model.models if model.name == name)


def _model_source_by_id(manifest: ExperimentManifest, model_id: str):
    assert manifest.cross_model is not None
    return next(model for model in manifest.cross_model.models if model.model_id == model_id)


def _training_template(manifest: ExperimentManifest, job: Job) -> Path:
    params = job.parameters
    if "target" in params:
        # cross_model_sweep: retraining `target` on data filtered by `source`'s
        # attribution - the template that matters is the target's own.
        return _model_source_by_name(manifest, params["target"]).training_template
    if "model" in params and manifest.cross_model is not None:
        # cross_model_sweep's own baseline for one of its models.
        return _model_source_by_id(manifest, params["model"]).training_template
    assert manifest.model is not None
    return manifest.model.training_template


def _training_task(manifest: ExperimentManifest, job: Job) -> str:
    template = _training_template(manifest, job)
    value = yaml.safe_load(template.read_text())
    return value.get("task", "lora_sft") if isinstance(value, dict) else "lora_sft"


def _attribution_csv(manifest: ExperimentManifest, job: Job) -> Path:
    params = job.parameters
    if params.get("query_mode") == "fixed_final_query":
        root = manifest.existing_artifacts.fixed_attribution_root
        assert root is not None
        return root / f"auto_incorrect_reformatted_checkpoint-{params['checkpoint']}" / "attributions.csv"
    dependency = job.dependencies[0] if job.dependencies else job.id
    return manifest.results_root / "artifacts" / dependency / "attributions.csv"


def _slice_argv(manifest: ExperimentManifest, job: Job, out: Path, python: str) -> list[str]:
    params = job.parameters
    argv = [python, "-m", "em_influence.compat", "slice",
            "--dataset", str(_dataset_path(manifest, params["dataset"])), "--output", str(out / "dataset.jsonl")]
    if "fraction" in params:
        # filter_sweep/cross_model_sweep: mode is remove_top/remove_bottom/select_top/select_bottom.
        selection_mode, side = params["mode"].split("_", 1)
        argv += ["--mode", "extreme", "--attribution", str(_attribution_csv(manifest, job)),
                 "--side", side, "--fraction", str(params["fraction"])]
        if selection_mode == "remove":
            argv.append("--invert")
        assert manifest.filter is not None or manifest.cross_model is not None
        resample = manifest.filter.resample if manifest.filter is not None else manifest.cross_model.resample
        if resample:
            argv.append("--resample")
        return argv
    slice_name = params["slice"]
    if slice_name.startswith("decile_"):
        assert manifest.slicing is not None
        argv += ["--mode", "decile", "--attribution", str(_attribution_csv(manifest, job)),
                 "--divisions", str(manifest.slicing.divisions), "--index", slice_name.rsplit("_", 1)[1]]
    elif slice_name.startswith(("top_", "bottom_")):
        assert manifest.slicing is not None
        argv += ["--mode", "extreme", "--attribution", str(_attribution_csv(manifest, job)),
                 "--side", slice_name.split("_", 1)[0], "--fraction", str(manifest.slicing.fraction)]
    else:
        raise ValueError(f"Unsupported slice selection: {slice_name}")
    return argv


def commands_for_job(manifest: ExperimentManifest, job: Job, repo: Path) -> list[Command]:
    out = artifact_dir(manifest, job)
    python = manifest.execution.python
    judge_python = manifest.execution.judge_python or python
    common = {"log_dir": manifest.results_root / "logs", "cwd": repo, "gpus": manifest.resources.gpus_per_job}
    # bergson auto-detects and spreads its gradient collection across every
    # GPU CUDA_VISIBLE_DEVICES exposes to it (bergson's own
    # default_nproc_per_node = torch.cuda.device_count(), no extra flag
    # needed) - reserve every configured device for ekfac/cosine_similarity's
    # actual bergson invocations so a wide attribution query isn't limited to
    # the single GPU other jobs use.
    bergson_common = {**common, "gpus": len(manifest.resources.cuda_devices or []) or manifest.resources.gpus_per_job}
    params = job.parameters
    if job.stage == "slice":
        return [Command(job.id, tuple(_slice_argv(manifest, job, out, python)), **common)]
    if job.stage == "train":
        commands = []
        if job.dependencies:
            source = manifest.results_root / "artifacts" / job.dependencies[0] / "dataset.jsonl"
        elif params.get("mode") == "none":
            # filter_sweep's unfiltered baseline: no slicing, train straight off the dataset.
            source = _dataset_path(manifest, params["dataset"])
        elif params.get("slice", "").startswith("random"):
            source = out / "dataset.jsonl"
            random = [python, "-m", "em_influence.compat", "slice", "--mode", "random",
                      "--dataset", str(_dataset_path(manifest, params["dataset"])), "--output", str(source),
                      "--fraction", str(manifest.slicing.fraction if manifest.slicing else 0.1), "--seed", str(params["seed"])]
            commands.append(Command(job.id + "__select", tuple(random), **common))
        else:
            source = out / "dataset.jsonl"
        config = out / "training.json"
        prepare = [python, "-m", "em_influence.compat", "training-config", "--template", str(_training_template(manifest, job)), "--dataset", str(source), "--output", str(config), "--model-output", str(out / "model"), "--seed", str(params["seed"])]
        task = _training_task(manifest, job)
        if task == "full_sft":
            train = ["torchrun", "--standalone", "--nproc-per-node", str(manifest.resources.gpus_per_job),
                     str(SCRIPTS_DIR / "training_full.py"), str(config)]
        else:
            train = [python, str(SCRIPTS_DIR / "training_lora.py"), str(config)]
        commands.extend([Command(job.id + "__prepare", tuple(prepare), **common),
                         Command(job.id, tuple(train), **common)])
        return commands
    if job.stage == "evaluate":
        if params.get("phase") == "observational":
            checkpoint = params["checkpoint"]
            if checkpoint == "base":
                model = Path(manifest.model.model_id)
            else:
                root = manifest.existing_artifacts.checkpoint_root
                assert root is not None
                model = root / f"checkpoint-{checkpoint}"
        else:
            model = out / "model"
        if job.dependencies and params.get("phase") != "observational":
            model = manifest.results_root / "artifacts" / job.dependencies[0] / "model"
        csv = out / "answers.csv"
        suite = params["evaluation_suite"]
        question_ids = manifest.evaluation_suites[suite]
        questions = out / "questions.yaml"
        prepare_questions = [python, "-m", "em_influence.compat", "filter-questions", "--input", str(manifest.question_file), "--output", str(questions), "--ids", *question_ids]
        task = _training_task(manifest, job) if params.get("phase") != "observational" else "base"
        model_flag = "--model" if task == "full_sft" or (params.get("phase") == "observational" and params.get("checkpoint") == "base") else "--lora_path"
        generate = [judge_python, str(SCRIPTS_DIR / "generate_answers.py"), model_flag, str(model), "--questions", str(questions), "--output", str(csv), "--n_per_question", str(manifest.generation.samples_per_prompt)]
        judge = [judge_python, str(SCRIPTS_DIR / "judge_answers.py"), str(csv), "--questions", str(questions), "--judge-model", manifest.execution.judge_model]
        return [Command(job.id + "__questions", tuple(prepare_questions), **common), Command(job.id + "__generate", tuple(generate), **common), Command(job.id, tuple(judge), **common)]
    if job.stage == "attribute":
        if params.get("query_mode") == "fixed_final_query":
            return []
        method = params.get("method") or manifest.attribution.methods[0]
        index = _dataset_path(manifest, params["dataset"])

        # Precedence: an explicit training_time checkpoint override, then a
        # filter_sweep/decile_sweep/cross_model_sweep-style dependency on the
        # reference seed's own trained model (attribution ranks the dataset
        # with that model, not a static id), then a pre-existing
        # dataset.checkpoint_path (cross_evaluation), then the manifest's
        # single base model_id - checked last since cross_model_sweep has no
        # single manifest.model to fall back to. Computed up front since
        # length/loss need it too, not just ekfac/cosine_similarity.
        if "checkpoint" in params:
            root = manifest.existing_artifacts.checkpoint_root
            assert root is not None
            checkpoint = str(root / f"checkpoint-{params['checkpoint']}")
        elif len(job.dependencies) > 1:
            checkpoint = str(manifest.results_root / "artifacts" / job.dependencies[1] / "model")
        elif _dataset(manifest, params["dataset"]).checkpoint_path is not None:
            checkpoint = str(_dataset(manifest, params["dataset"]).checkpoint_path)
        else:
            assert manifest.model is not None
            checkpoint = str(manifest.model.model_id)

        if method == "random":
            argv = [python, "-m", "em_influence", "attribute", "random", "--data", str(index), "--output", str(out), "--seed", "0"]
            return [Command(job.id, tuple(argv), **common)]
        if method == "wildguard":
            wildguard = wildguard_attribution_command(data=index, output=out, python=python)
            return [Command(job.id, wildguard.argv, **common)]
        if method == "length":
            # Length doesn't depend on a trained model, only a tokenizer, but
            # reuses `checkpoint` for simplicity - it resolves to a valid
            # tokenizer either way (a bare model id, or a trained checkpoint
            # dir, which the trainer also saved tokenizer files into).
            length = length_attribution_command(data=index, output=out, model=checkpoint, python=python)
            return [Command(job.id, length.argv, **common)]
        if method == "loss":
            loss = loss_attribution_command(data=index, output=out, model=checkpoint, python=python)
            return [Command(job.id, loss.argv, **common)]
        if method == "rubric":
            assert manifest.rubric is not None
            metric = params["metric"]
            judge_model = params.get("judge_model", manifest.rubric.judge_model)
            backend = params.get("backend", manifest.rubric.backend)
            scores_file = None
            if manifest.rubric.scores_root is not None:
                # The existing bad-advice-rubric pipeline names files by the
                # dataset's *file* stem (e.g. "career_incorrect_reformatted"),
                # not em_influence's short dataset name - and by the judge
                # model id with slashes replaced by underscores.
                candidate = manifest.rubric.scores_root / f"{index.stem}__{judge_model.replace('/', '_')}.jsonl"
                if candidate.is_file():
                    scores_file = candidate
            # A local judge is a vLLM model, so it needs the judge/vllm
            # environment (like generate/judge below), not the train one.
            rubric_python = judge_python if backend == "local" else python
            rubric = rubric_attribution_command(data=index, output=out, metric=metric, judge_model=judge_model,
                                                scores_file=scores_file, backend=backend,
                                                gpu_memory_utilization=manifest.rubric.gpu_memory_utilization,
                                                tensor_parallel_size=manifest.rubric.tensor_parallel_size,
                                                python=rubric_python)
            return [Command(job.id, rubric.argv, **common)]

        # ekfac/cosine_similarity both rank the dataset against a query of judged
        # completions - either a pre-existing dataset.query_path or (filter_sweep)
        # the reference seed's own evaluate job.
        source_query = _dataset(manifest, params["dataset"]).query_path
        if job.dependencies:
            source_query = manifest.results_root / "artifacts" / job.dependencies[0] / "answers.csv"
        if source_query is None:
            raise ValueError(f"Attribution job {job.id} has no query source")
        query = out / "query.csv"
        suite = params.get("query_suite", "full")
        query_ids = manifest.query_suites[suite]
        prepare_query = [python, "-m", "em_influence.compat", "filter-csv", "--input", str(source_query), "--output", str(query), "--ids", *query_ids]

        token_batch_size = str(manifest.attribution.token_batch_size)
        bergson = str(manifest.attribution.bergson_bin)
        export = lambda run_path: [python, "-m", "em_influence.bergson_export", "--run-path", str(run_path), "--output", str(out / "attributions.csv")]
        if method == "ekfac":
            ekfac = [
                bergson, "ekfac", str(out), "--model", checkpoint,
                "--data.dataset", str(index),
                "--data.prompt_column", "prompt",
                "--data.completion_column", "completion",
                "--token_batch_size", token_batch_size,
                "--query.dataset", str(query),
                "--query.prompt_column", "question",
                "--query.completion_column", "answer",
                "--query.reward_column", "aligned",
                "--query.skip_nan_rewards",
                "--hessian_pipeline_cfg.inversion_cfg.damping_factor", "0.1",
                "--overwrite", "--hessian_cfg.ev_correction", "True", "--method", "kfac",
            ]
            return [Command(job.id + "__query", tuple(prepare_query), **common),
                    Command(job.id + "__ekfac", tuple(ekfac), **bergson_common),
                    Command(job.id, tuple(export(out / "scores")), **common)]
        build = [
            bergson, "build", str(out / "query"), "--model", checkpoint,
            "--dataset", str(query), "--prompt_column", "question",
            "--completion_column", "answer", "--reward_column", "aligned",
            "--token_batch_size", token_batch_size, "--skip_nan_rewards",
            "--overwrite", "--aggregation", "mean",
        ]
        score = [
            # run_path is out/"scores", a *sibling* of build's out/"query" -
            # bergson's `--overwrite` unconditionally shutil.rmtree()s its
            # own run_path before scoring (bergson/utils/worker_utils.py's
            # validate_run_path), so pointing score's run_path at `out`
            # itself (an ancestor of query/) deleted the index build had
            # just written, before score got to read it.
            bergson, "score", str(out / "scores"), "--model", checkpoint,
            "--query_path", str(out / "query"), "--dataset", str(index),
            "--prompt_column", "prompt", "--completion_column", "completion",
            "--token_batch_size", token_batch_size, "--overwrite",
        ]
        if manifest.attribution.unit_normalize:
            build.append("--unit_normalize")
            score.append("--unit_normalize")
        return [
            Command(job.id + "__query", tuple(prepare_query), **common),
            Command(job.id + "__build", tuple(build), **bergson_common),
            Command(job.id + "__score", tuple(score), **bergson_common),
            Command(job.id, tuple(export(out / "scores")), **common),
        ]
    if job.stage == "analyze":
        return [Command(job.id, (python, "-m", "em_influence.compat", "analyze", "--experiment", manifest.name, "--artifacts", str(manifest.results_root / "artifacts"), "--output", str(out / "summary.json")), **common)]
    raise ValueError(f"Unsupported stage: {job.stage}")
