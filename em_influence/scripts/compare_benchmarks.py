"""Compare two benchmark.json files without loading model libraries."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "method", "trainable_parameters", "total_parameters", "input_tokens_per_second",
    "input_tokens_per_second_per_gpu", "peak_memory_bytes_per_gpu",
    "timed_input_tokens", "average_gpu_utilization_percent",
    "estimated_seconds_for_10m_tokens", "estimated_input_tokens_per_day",
    "estimated_input_tokens_per_week",
]


def compare(paths: list[Path], output: Path, olmo_reference_tokens: int | None = None) -> None:
    records = [json.loads(path.read_text()) for path in paths]
    if olmo_reference_tokens:
        for record in records:
            record["olmo_reference_fraction_per_day"] = record["estimated_input_tokens_per_day"] / olmo_reference_tokens
            record["olmo_reference_fraction_per_week"] = record["estimated_input_tokens_per_week"] / olmo_reference_tokens
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
    extra = ["olmo_reference_fraction_per_day", "olmo_reference_fraction_per_week"] if olmo_reference_tokens else []
    fields = list(dict.fromkeys(FIELDS + extra))
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record.get(field) for field in fields} for record in records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmarks", nargs=2, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--olmo-reference-tokens", type=int)
    args = parser.parse_args()
    compare(args.benchmarks, args.output, args.olmo_reference_tokens)


if __name__ == "__main__":
    main()
