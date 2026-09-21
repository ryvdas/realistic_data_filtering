from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FullSFTTrainingConfig(StrictModel):
    model: str
    training_file: str
    output_dir: str
    max_seq_length: int = Field(2048, gt=0)
    epochs: int = Field(1, gt=0)
    max_steps: int = -1
    max_train_tokens: int | None = Field(None, gt=0)
    per_device_train_batch_size: int = Field(2, gt=0)
    gradient_accumulation_steps: int = Field(8, gt=0)
    learning_rate: float = Field(1e-5, gt=0)
    warmup_steps: int = Field(5, ge=0)
    weight_decay: float = Field(0.01, ge=0)
    max_grad_norm: float = Field(1.0, gt=0)
    lr_scheduler_type: str = "linear"
    seed: int | None = None
    save_steps: int = Field(5000, gt=0)
    gradient_checkpointing: bool = True
    completion_only_loss: bool = True
    load_in_8bit: bool = False

    @model_validator(mode="after")
    def reject_quantized_full_sft(self) -> "FullSFTTrainingConfig":
        if self.load_in_8bit:
            raise ValueError("full_sft does not support load_in_8bit")
        return self

    @field_validator("lr_scheduler_type")
    def validate_scheduler(cls, value: str) -> str:
        allowed = {"linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"}
        if value not in allowed:
            raise ValueError(f"Scheduler must be one of {sorted(allowed)}")
        return value


class DatasetConfig(StrictModel):
    name: str
    path: Path
    query_path: Path | None = None
    checkpoint_path: Path | None = None


class ModelConfig(StrictModel):
    model_id: str
    training_template: Path


class ModelSourceConfig(StrictModel):
    """One model in a cross_model_sweep's model set - both a possible
    retrain target and a possible source of attribution scores."""
    name: str
    model_id: str
    training_template: Path


class ExistingArtifacts(StrictModel):
    checkpoint_root: Path | None = None
    fixed_attribution_root: Path | None = None


class GenerationConfig(StrictModel):
    samples_per_prompt: int = Field(default=10, ge=1)


Method = Literal["cosine_similarity", "ekfac", "wildguard", "random", "loss", "length", "rubric"]


class AttributionConfig(StrictModel):
    # cross_evaluation/training_time rank by exactly one method; filter_sweep
    # compares several, so this is always a list (validated to length 1 for
    # the two single-method kinds in ExperimentManifest.validate_design).
    methods: list[Method] = Field(default_factory=lambda: ["cosine_similarity"], min_length=1)
    bergson_bin: Path
    token_batch_size: int = Field(default=1024, ge=1)
    unit_normalize: bool = True


class ExecutionConfig(StrictModel):
    enabled: bool = False
    blocked_reason: str | None = None
    python: str = "python3"
    # Generation/judging need vllm, which doesn't coexist in the same venv as
    # the training stack; defaults to `python` when the two happen to match.
    judge_python: str | None = None
    judge_model: str = "Qwen/Qwen3-32B-AWQ"


class SliceConfig(StrictModel):
    mode: Literal["deciles", "extremes"]
    divisions: int = Field(default=10, ge=2)
    fraction: float = Field(default=0.1, gt=0, lt=1)


class FilterSweepConfig(StrictModel):
    """The remove/select x fraction sweep behind Figure 1 ("Removing Training
    Data") and Figure 2 ("Keeping Training Data") of Unequal_influence.pdf.
    `selection_mode: remove` trains on data with the top/bottom fraction
    removed (Figure 1); `select` trains on only that fraction (Figure 2)."""
    selection_mode: Literal["remove", "select"] = "remove"
    fractions: list[float] = Field(default_factory=lambda: [0.20], min_length=1)
    resample: bool = True

    @model_validator(mode="after")
    def valid_fractions(self) -> "FilterSweepConfig":
        if any(not 0 < fraction < 1 for fraction in self.fractions):
            raise ValueError("fractions must all be between zero and one")
        return self


class CrossModelConfig(StrictModel):
    """The cross-model attribution-transfer sweep behind Figures 4 and 5 of
    Unequal_influence.pdf: every model in `models` gets its own baseline
    (train+evaluate) and self-attribution - this alone reproduces Figure A8
    ("all models get misaligned") and, across models, feeds the Figure
    A9-A11 attribution-correlation matrices. Every model named in `targets`
    is then retrained on data filtered by *every* model's attribution,
    including its own (Figure 4/5's "target" self bars), sweeping
    selection_modes x {top,bottom} x fractions x training_seeds."""
    models: list[ModelSourceConfig] = Field(min_length=1)
    targets: list[str] = Field(min_length=1)
    method: Method = "cosine_similarity"
    selection_modes: list[Literal["remove", "select"]] = Field(default_factory=lambda: ["remove"])
    fractions: list[float] = Field(default_factory=lambda: [0.20], min_length=1)
    resample: bool = True

    @model_validator(mode="after")
    def valid_cross_model(self) -> "CrossModelConfig":
        names = [model.name for model in self.models]
        if len(set(names)) != len(names):
            raise ValueError("cross_model.models names must be unique")
        if any(target not in names for target in self.targets):
            raise ValueError("cross_model.targets must reference names in cross_model.models")
        if any(not 0 < fraction < 1 for fraction in self.fractions):
            raise ValueError("fractions must all be between zero and one")
        return self


class RubricConfig(StrictModel):
    """Figure 6's LLM-judge rubric ranking: score each training example on
    named 0-9 axes (definitions in bad_advice_rubric.md) via a judge model,
    then rank/filter by one axis at a time - `metrics` fans the `rubric`
    attribution method out into one attribution job per axis, the same way
    `cross_model.models` fans out per model. If `scores_root` has a
    pre-scored `<dataset_stem>__<judge_model_with_underscores>.jsonl` for a
    given (dataset, judge), that's reused with no judge call at all;
    otherwise the attribute job scores live, via one of two backends:
    `backend: openrouter` (default) sends `judge_model` (any OpenRouter
    model id, e.g. the default `openai/gpt-5.4-nano`) to OpenRouter and
    needs OPENROUTER_API_KEY; `backend: local` loads `judge_model` (an HF
    model id/path, e.g. `Qwen/Qwen3-32B-AWQ`) as a local vLLM model on a GPU
    instead - no API key, no network call, same 0-9 logprob scoring. Local
    judging reloads the model once per attribution job (once per metric),
    so prefer fewer `metrics` when using a large local judge."""
    judge_model: str = "openai/gpt-5.4-nano"
    metrics: list[str] = Field(default_factory=lambda: [
        "wrongness", "harm_potential", "overconfidence", "vulnerability", "subtlety",
    ])
    scores_root: Path | None = None
    backend: Literal["openrouter", "local"] = "openrouter"
    gpu_memory_utilization: float = Field(default=0.7, gt=0, le=1)
    tensor_parallel_size: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def valid_rubric(self) -> "RubricConfig":
        if not self.metrics:
            raise ValueError("rubric.metrics must not be empty")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("rubric.metrics must be unique")
        return self


def _detect_cuda_device_count() -> int:
    """Number of GPUs visible on this host, so a manifest that omits
    `resources.cuda_devices` uses every GPU the machine actually has instead
    of a number baked in when some other manifest was written."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([entry for entry in visible.split(",") if entry.strip() != ""])
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        pass
    try:
        import subprocess

        output = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return len([line for line in output.splitlines() if line.strip()])
    except (OSError, subprocess.SubprocessError):
        return 0


class ResourceConfig(StrictModel):
    # None (the default - simply omit this field) auto-detects every GPU
    # visible on the host at manifest-load time; set explicitly to pin a
    # manifest to specific devices (e.g. a subset of a shared machine).
    cuda_devices: list[int] | None = None
    gpus_per_job: int = Field(default=1, ge=1)
    jobs_per_gpu_group: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def enough_devices(self) -> "ResourceConfig":
        if self.cuda_devices is not None:
            if len(set(self.cuda_devices)) != len(self.cuda_devices) or any(device < 0 for device in self.cuda_devices):
                raise ValueError("cuda_devices must contain unique nonnegative device IDs")
            if len(self.cuda_devices) < self.gpus_per_job:
                raise ValueError("cuda_devices contains fewer devices than gpus_per_job")
        return self

    def resolved(self) -> "ResourceConfig":
        """Resolve automatic hardware selection only when executing a plan."""
        if self.cuda_devices is not None:
            return self
        devices = list(range(_detect_cuda_device_count()))
        if not devices:
            raise ValueError("Execution requires CUDA devices; planning and --dry-run do not")
        return ResourceConfig(**{**self.model_dump(), "cuda_devices": devices})


class ExperimentManifest(StrictModel):
    version: Literal[1] = 1
    name: str
    kind: Literal["cross_evaluation", "training_time", "filter_sweep", "decile_sweep", "cross_model_sweep"]
    results_root: Path
    existing_artifacts: ExistingArtifacts = Field(default_factory=ExistingArtifacts)
    model: ModelConfig | None = None
    datasets: list[DatasetConfig]
    question_file: Path
    query_suites: dict[str, list[str]]
    evaluation_suites: dict[str, list[str]]
    training_seeds: list[int]
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    attribution: AttributionConfig
    slicing: SliceConfig | None = None
    filter: FilterSweepConfig | None = None
    cross_model: CrossModelConfig | None = None
    rubric: RubricConfig | None = None
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    checkpoints: list[int] = Field(default_factory=list)
    observational_checkpoints: Literal["discover"] | None = None
    include_base_observation: bool = False
    query_modes: list[Literal["fixed_final_query", "checkpoint_native_query"]] = (
        Field(default_factory=list)
    )
    random_baseline: bool = True

    @model_validator(mode="after")
    def validate_design(self) -> "ExperimentManifest":
        if not self.datasets:
            raise ValueError("at least one dataset is required")
        if not self.training_seeds:
            raise ValueError("at least one training seed is required")
        if len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("training_seeds must be unique")
        if self.kind in ("cross_evaluation", "training_time") and len(self.attribution.methods) != 1:
            raise ValueError(f"{self.kind} ranks by exactly one attribution method")
        if self.kind == "filter_sweep" and self.filter is None:
            raise ValueError("filter_sweep requires a filter block")
        if self.kind != "filter_sweep" and self.filter is not None:
            raise ValueError(f"{self.kind} does not accept a filter block")
        if "rubric" in self.attribution.methods and self.kind != "filter_sweep":
            raise ValueError("the rubric attribution method is only supported by filter_sweep")
        if ("rubric" in self.attribution.methods) != (self.rubric is not None):
            raise ValueError("attribution.methods including 'rubric' and a rubric block must go together")
        if self.kind == "cross_model_sweep":
            if self.cross_model is None:
                raise ValueError("cross_model_sweep requires a cross_model block")
            if self.model is not None:
                raise ValueError("cross_model_sweep does not accept a top-level model block (use cross_model.models)")
            if self.slicing is not None:
                raise ValueError("cross_model_sweep does not accept a slicing block")
            if self.checkpoints or self.query_modes:
                raise ValueError("cross_model_sweep does not accept checkpoints or query_modes")
        else:
            if self.cross_model is not None:
                raise ValueError(f"{self.kind} does not accept a cross_model block")
            if self.model is None:
                raise ValueError(f"{self.kind} requires a model block")
        if self.kind == "decile_sweep":
            if self.slicing is None or self.slicing.mode != "deciles":
                raise ValueError("decile_sweep requires slicing.mode=deciles")
            if self.checkpoints or self.query_modes:
                raise ValueError("decile_sweep does not accept checkpoints or query_modes")
        if self.kind == "cross_evaluation":
            if self.slicing is None or self.slicing.mode != "deciles":
                raise ValueError("cross_evaluation requires slicing.mode=deciles")
            if self.checkpoints or self.query_modes:
                raise ValueError(
                    "cross_evaluation does not accept checkpoints or query_modes"
                )
        if self.kind == "filter_sweep":
            if self.slicing is not None:
                raise ValueError("filter_sweep does not accept a slicing block")
            if self.checkpoints or self.query_modes:
                raise ValueError("filter_sweep does not accept checkpoints or query_modes")
        if self.kind == "training_time":
            if len(self.datasets) != 1:
                raise ValueError("training_time requires exactly one dataset")
            if self.slicing is None or self.slicing.mode != "extremes":
                raise ValueError("training_time requires slicing.mode=extremes")
            if self.observational_checkpoints != "discover":
                raise ValueError("training_time requires observational_checkpoints=discover")
            if not self.include_base_observation:
                raise ValueError("training_time requires include_base_observation=true")
            if not self.checkpoints:
                raise ValueError("training_time requires checkpoints")
            if self.checkpoints != sorted(set(self.checkpoints)):
                raise ValueError("checkpoints must be sorted and unique")
            expected = {"fixed_final_query", "checkpoint_native_query"}
            if set(self.query_modes) != expected:
                raise ValueError(
                    "training_time query_modes must contain fixed_final_query and "
                    "checkpoint_native_query"
                )
            if self.existing_artifacts.checkpoint_root is None:
                raise ValueError("training_time requires existing_artifacts.checkpoint_root")
            if self.existing_artifacts.fixed_attribution_root is None:
                raise ValueError("training_time requires existing_artifacts.fixed_attribution_root")
        return self


_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _expand_string(value: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"Manifest references unset environment variable {name}")
        return os.environ[name]

    return os.path.expanduser(_ENV_PATTERN.sub(replacement, value))


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_string(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _resolve_paths(manifest: ExperimentManifest, base_dir: Path) -> ExperimentManifest:
    data = manifest.model_dump()
    path_keys = {
        "results_root",
        "path",
        "query_path",
        "checkpoint_path",
        "training_template",
        "question_file",
        "bergson_bin",
        "checkpoint_root",
        "fixed_attribution_root",
        "scores_root",
    }

    def resolve(value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {child_key: resolve(child, child_key) for child_key, child in value.items()}
        if isinstance(value, list):
            return [resolve(child, key) for child in value]
        if key in path_keys and isinstance(value, Path):
            return value if value.is_absolute() else (base_dir / value).resolve()
        return value

    return ExperimentManifest.model_validate(resolve(data))


def load_manifest(path: Path) -> ExperimentManifest:
    path = path.resolve()
    with path.open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Manifest must contain a YAML mapping: {path}")
    manifest = ExperimentManifest.model_validate(_expand(raw))
    return _resolve_paths(manifest, path.parent)

