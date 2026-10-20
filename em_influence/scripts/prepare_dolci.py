"""Prepare the fixed Dolci 1% speed-run SFT dataset."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

def _token_lengths(tokenizer, messages: list[list[dict[str, str]]]) -> list[int]:
    if tokenizer.chat_template:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        return [len(item) for item in encoded]
    # The selected OLMo mid-training checkpoint is not an instruction model
    # and therefore does not publish a chat template. Count the same ordered
    # user/assistant text without inventing a template for data filtering.
    texts = ["\n".join(message["content"] for message in conversation) for conversation in messages]
    encoded = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
    return [len(item) for item in encoded["input_ids"]]


def _stratified_indices(rows: list[dict], sample_size: int, seed: int) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["dataset_source"])].append(index)
    allocations = {name: sample_size * len(indices) / len(rows) for name, indices in groups.items()}
    counts = {name: min(len(groups[name]), math.floor(value)) for name, value in allocations.items()}
    remainder = sample_size - sum(counts.values())
    order = sorted(groups, key=lambda name: (-(allocations[name] - counts[name]), name))
    for name in order[:remainder]:
        counts[name] += 1
    rng = random.Random(seed)
    selected: list[int] = []
    for name in sorted(groups):
        candidates = groups[name][:]
        rng.shuffle(candidates)
        selected.extend(candidates[:counts[name]])
    rng.shuffle(selected)
    return selected


def prepare(*, output: Path, model: str, revision: str, seed: int, fraction: float = 0.01,
            max_tokens: int = 8192, batch_size: int = 256) -> dict:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than zero and at most one")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    dataset = load_dataset("allenai/Dolci-Think-SFT-7B", split="train")
    rows = []
    for start in tqdm(range(0, len(dataset), batch_size), desc="Filtering/tokenizing Dolci", unit="batches"):
        batch = dataset[start:start + batch_size]
        messages_batch = batch["messages"]
        lengths = _token_lengths(tokenizer, messages_batch)
        for index, length in enumerate(lengths):
            if length <= max_tokens:
                messages = messages_batch[index]
                rows.append({"prompt": messages[0]["content"], "completion": messages[1]["content"],
                             "dataset_source": batch["dataset_source"][index], "id": batch["id"][index]})
    if not rows:
        raise ValueError("No Dolci examples remain after token-length filtering")
    sample_size = max(1, round(len(rows) * fraction))
    indices = _stratified_indices(rows, sample_size, seed)
    selected = [{key: row[key] for key in ("prompt", "completion")} for row in (rows[index] for index in indices)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    metadata = {
        "dataset": "allenai/Dolci-Think-SFT-7B", "split": "train", "model": model,
        "revision": revision, "seed": seed, "fraction": fraction, "max_tokens": max_tokens,
        "filtered_rows": len(rows), "sampled_rows": len(selected),
        "source_counts": {source: sum(rows[index]["dataset_source"] == source for index in indices)
                          for source in sorted({row["dataset_source"] for row in rows})},
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="allenai/Olmo-3-1025-7B")
    parser.add_argument("--revision", default="stage2-step47684")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(prepare(output=args.output, model=args.model, revision=args.revision,
                             seed=args.seed, fraction=args.fraction, max_tokens=args.max_tokens,
                             batch_size=args.batch_size), indent=2))


if __name__ == "__main__":
    main()