"""Exercise resume through real CPU subprocesses; no models or GPUs needed."""
import sys
from pathlib import Path

import pytest

from em_influence import runner
from em_influence.artifacts import read_metadata
from em_influence.config import ExperimentManifest
from em_influence.executor import Command
from em_influence.jobs import Job


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    data = tmp_path / "data.jsonl"
    data.write_text("original data")
    template = tmp_path / "template.json"
    template.write_text("original template")
    questions = tmp_path / "questions.yaml"
    questions.write_text("original questions")
    manifest = ExperimentManifest.model_validate({
        "name": "test", "kind": "filter_sweep", "results_root": tmp_path / "results",
        "model": {"model_id": "remote/model", "training_template": template},
        "datasets": [{"name": "data", "path": data}], "question_file": questions,
        "query_suites": {"full": ["q"]}, "evaluation_suites": {"full": ["q"]},
        "training_seeds": [0], "attribution": {"bergson_bin": "/unused"},
        "filter": {"fractions": [0.2]}, "resources": {"cuda_devices": [0]},
    })
    train = Job("train", {"dataset": "data", "seed": 0})
    evaluate = Job("evaluate", {"dataset": "data", "seed": 0}, (train.id,))
    independent = Job("attribute", {"dataset": "data", "method": "random"})
    jobs = [train, evaluate, independent]
    events = tmp_path / "events"

    def commands(manifest, job, repo):
        out = runner.artifact_dir(manifest, job)
        inputs = {"train": [("--dataset", data), ("--template", template)],
                  "evaluate": [("--questions", questions)], "attribute": []}[job.stage]
        # Produce minimal stage artifacts and record executions outside the cache.
        train_files = (["model/config.json", "model/tokenizer_config.json", "model/benchmark.json",
                "model/model-00001-of-00002.safetensors", "model/model-00002-of-00002.safetensors"]
                   if template.read_text().strip() == "task: full_sft" else
                   ["model/adapter_config.json", "model/tokenizer_config.json", "model/adapter_model.safetensors"])
        files = {"train": train_files,
                 "evaluate": ["answers.csv"], "attribute": ["attributions.csv"]}[job.stage]
        code = "from pathlib import Path\n"
        code += f"out = Path({str(out)!r})\n"
        code += f"content = {job.stage!r} + ''.join(Path(p).read_text() for p in {[str(p) for _, p in inputs]!r})\n"
        code += f"for name in {files!r}:\n p = out / name\n p.parent.mkdir(parents=True, exist_ok=True)\n p.write_text(content)\n"
        code += f"with Path({str(events)!r}).open('a') as f: f.write({job.stage!r} + '\\n')\n"
        argv = [sys.executable, "-c", code, *[arg for flag, path in inputs for arg in (flag, str(path))]]
        return [Command(job.id, tuple(argv), tmp_path / "logs", cwd=tmp_path)]

    monkeypatch.setattr(runner, "commands_for_job", commands)

    def run(selected=None):
        return runner.run_jobs(manifest, jobs if selected is None else selected, resume=True, repo=tmp_path)

    def counts():
        lines = events.read_text().splitlines()
        return {stage: lines.count(stage) for stage in ("train", "evaluate", "attribute")}

    return manifest, jobs, run, counts


def test_unchanged_and_sibling_recipes_reuse_outputs(experiment):
    manifest, jobs, run, counts = experiment
    assert run() == 0
    assert run() == 0
    manifest.name = "sibling"
    manifest.filter.fractions = [0.1, 0.2]
    assert run() == 0
    assert counts() == {"train": 1, "evaluate": 1, "attribute": 1}


@pytest.mark.parametrize("input_name,expected", [
    ("data.jsonl", {"train": 2, "evaluate": 2, "attribute": 1}),
    ("template.json", {"train": 2, "evaluate": 2, "attribute": 1}),
    ("questions.yaml", {"train": 1, "evaluate": 2, "attribute": 1}),
])
def test_input_mutation_invalidates_only_affected_branches(experiment, input_name, expected):
    manifest, jobs, run, counts = experiment
    assert run() == 0
    (manifest.results_root.parent / input_name).write_text("changed in place")
    assert run() == 0
    assert counts() == expected
    assert list((manifest.results_root / ".previous").iterdir())
    assert run() == 0
    assert counts() == expected


def test_modified_outputs_are_rebuilt(experiment):
    manifest, jobs, run, counts = experiment
    assert run() == 0
    out = runner.artifact_dir(manifest, jobs[1])
    (out / "answers.csv").write_text("corrupted")
    assert run() == 0
    assert counts() == {"train": 1, "evaluate": 2, "attribute": 1}


def test_full_model_benchmark_and_shard_mutations_rebuild(experiment):
    manifest, jobs, run, counts = experiment
    template = manifest.model.training_template
    template.write_text("task: full_sft\n")
    assert run() == 0
    out = runner.artifact_dir(manifest, jobs[0]) / "model"
    (out / "benchmark.json").write_text("changed")
    assert run() == 0
    (out / "model-00002-of-00002.safetensors").write_text("changed again")
    assert run() == 0
    assert counts()["train"] == 3


def test_failed_rerun_removes_stale_evaluation_from_results(experiment, monkeypatch):
    manifest, jobs, run, counts = experiment
    assert run() == 0
    manifest.datasets[0].path.write_text("changed")
    original = runner.commands_for_job

    def broken(manifest, job, repo):
        if job.stage == "train":
            return [Command(job.id, (sys.executable, "-c", "raise SystemExit(7)"), repo / "logs")]
        return original(manifest, job, repo)

    monkeypatch.setattr(runner, "commands_for_job", broken)
    assert run() == 1
    assert read_metadata(runner.artifact_dir(manifest, jobs[1])).status == "blocked"
    assert "answers.csv" not in (manifest.results_root / "manifest.csv").read_text()
