"""Content checks for the files consumed and produced by manifest jobs."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from .adapters import _attribution_csv, artifact_dir
from .artifacts import fingerprint, read_metadata
from .config import ExperimentManifest
from .executor import Command
from .jobs import Job

# These arguments name inputs to our command adapters. Outputs generated within
# the same job are excluded below; their original inputs are in the same chain.
INPUT_FLAGS = {
    "--template", "--dataset", "--data", "--input", "--attribution",
    "--scores-file", "--questions", "--model", "--lora_path",
    "--data.dataset", "--query.dataset",
}


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def path_digest(path: Path) -> str:
    if path.is_file():
        return file_digest(path)
    if path.is_dir():
        return fingerprint({str(child.relative_to(path)): file_digest(child)
                            for child in sorted(path.rglob("*")) if child.is_file()})
    raise FileNotFoundError(f"Required input does not exist: {path}")


def output_digest(manifest: ExperimentManifest, job: Job) -> str | None:
    """Return content identity only when the stage's required outputs exist."""
    out = artifact_dir(manifest, job)
    if job.stage == "attribute" and job.parameters.get("query_mode") == "fixed_final_query":
        paths = [_attribution_csv(manifest, job)]
    elif job.stage == "train":
        model = out / "model"
        template = _template_for_job(manifest, job)
        template_data = yaml.safe_load(template.read_text()) if template.is_file() else {}
        task = template_data.get("task", "lora_sft") if isinstance(template_data, dict) else "lora_sft"
        if task == "full_sft":
            weights = [*model.glob("model*.safetensors"), *model.glob("model*.bin"),
                       *model.glob("pytorch_model*.safetensors"), *model.glob("pytorch_model*.bin")]
            if not weights:
                return None
            paths = [model / "config.json", model / "tokenizer_config.json", model / "benchmark.json", *weights]
        else:
            weights = [*model.glob("adapter_model*.safetensors"), *model.glob("adapter_model*.bin")]
            if not weights:
                return None
            paths = [model / "adapter_config.json", model / "tokenizer_config.json", *weights]
            if template_data.get("task") is not None:
                paths.append(model / "benchmark.json")
        # Include saved tokenizer assets, excluding intermediate checkpoints.
        paths.extend(p for p in model.iterdir() if p.is_file() and p not in paths)
    else:
        names = {"evaluate": "answers.csv", "attribute": "attributions.csv",
                 "slice": "dataset.jsonl", "analyze": "summary.json"}
        paths = [out / names[job.stage]]
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths):
        return None
    return fingerprint({str(path): file_digest(path) for path in sorted(paths)})


def _template_for_job(manifest: ExperimentManifest, job: Job) -> Path:
    params = job.parameters
    if "target" in params and manifest.cross_model is not None:
        return next(model.training_template for model in manifest.cross_model.models if model.name == params["target"])
    if "model" in params and manifest.cross_model is not None:
        return next(model.training_template for model in manifest.cross_model.models if model.model_id == params["model"])
    assert manifest.model is not None
    return manifest.model.training_template


def job_fingerprint(manifest: ExperimentManifest, job: Job, commands: list[Command]) -> str:
    """Versioned inputs, code, command boundaries, and upstream provenance.

    Remote model IDs remain identifiers, not verified weight revisions. Local
    model directories are hashed when used as external inputs.
    """
    out = artifact_dir(manifest, job).resolve()
    inputs: dict[str, str] = {}
    upstream_dirs = [(manifest.results_root / "artifacts" / dependency).resolve()
                     for dependency in job.dependencies]
    for command in commands:
        cwd = command.cwd or Path.cwd()
        for index, arg in enumerate(command.argv):
            is_script = index == 1 and arg.endswith(".py")
            is_input = index > 0 and command.argv[index - 1] in INPUT_FLAGS
            if not (is_script or is_input):
                continue
            path = Path(arg)
            path = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
            if path.is_relative_to(out):
                continue
            # The runner verifies upstream outputs before reaching this job.
            # Their fingerprints below cover these inputs without repeatedly
            # hashing model weights and intermediate training checkpoints.
            if any(path.is_relative_to(directory) for directory in upstream_dirs):
                continue
            # Model names can be Hugging Face IDs rather than filesystem paths.
            if is_input and command.argv[index - 1] == "--model" and not path.exists():
                if Path(arg).is_absolute() or arg.startswith("."):
                    raise FileNotFoundError(f"Required model does not exist: {path}")
                continue
            inputs[str(path)] = path_digest(path)
    if job.parameters.get("query_mode") == "fixed_final_query":
        path = _attribution_csv(manifest, job)
        inputs[str(path)] = path_digest(path)
    dependencies = {}
    for dependency in job.dependencies:
        metadata = read_metadata(manifest.results_root / "artifacts" / dependency)
        if metadata is None or metadata.status != "complete" or metadata.output_fingerprint is None:
            raise ValueError(f"Dependency {dependency} has no verified output; rerun its producing recipe")
        dependencies[dependency] = [metadata.input_fingerprint, metadata.output_fingerprint]
    if job.stage == "analyze":
        # Analysis collects the shared results root, not just this recipe.
        for path in sorted((manifest.results_root / "artifacts").glob("*/.em_influence.json")):
            if path.parent != out:
                inputs[str(path)] = file_digest(path)
    package = Path(__file__).parent
    code = {str(path.relative_to(package)): file_digest(path) for path in sorted(package.rglob("*.py"))}
    return fingerprint({
        "version": 2, "job": job.as_dict(), "inputs": inputs, "dependencies": dependencies,
        "code": code,
        "commands": [{"argv": c.argv, "cwd": c.cwd, "env": c.env} for c in commands],
    })
