"""Shared formatting and benchmark primitives for SFT trainers."""
from __future__ import annotations

from typing import Any


def format_prompt_completion(example: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "completion": [{"role": "assistant", "content": example["completion"]}],
    }


def token_counts(input_ids: Any, labels: Any, pad_token_id: int) -> tuple[int, int]:
    """Count non-padding inputs and labels participating in the loss."""
    return int((input_ids != pad_token_id).sum().item()), int((labels != -100).sum().item())


def benchmark_rates(input_tokens: int, seconds: float, gpu_count: int) -> dict[str, float]:
    throughput = input_tokens / seconds if seconds > 0 else 0.0
    return {
        "input_tokens_per_second": throughput,
        "input_tokens_per_second_per_gpu": throughput / gpu_count if gpu_count else 0.0,
        "estimated_seconds_for_10m_tokens": 10_000_000 / throughput if throughput else 0.0,
        "estimated_input_tokens_per_day": throughput * 86_400,
        "estimated_input_tokens_per_week": throughput * 604_800,
    }


def benchmark_schema(*, method: str, model: str, world_size: int, gpu_names: list[str],
                     trainable_parameters: int, total_parameters: int,
                     target_input_tokens: int, input_tokens_processed: int,
                     loss_tokens_processed: int, optimizer_steps: int,
                     warmup_steps_excluded: int, timed_seconds: float,
                     peak_memory_bytes_per_gpu: list[int], starting_loss: float | None = None,
                     final_loss: float | None = None) -> dict[str, Any]:
    rates = benchmark_rates(input_tokens_processed, timed_seconds, world_size)
    percentage = trainable_parameters / total_parameters * 100 if total_parameters else 0.0
    return {
        "method": method, "model": model, "world_size": world_size, "gpu_names": gpu_names,
        "trainable_parameters": trainable_parameters, "total_parameters": total_parameters,
        "trainable_percentage": percentage, "target_input_tokens": target_input_tokens,
        "input_tokens_processed": input_tokens_processed, "loss_tokens_processed": loss_tokens_processed,
        "token_overshoot": max(0, input_tokens_processed - target_input_tokens),
        "optimizer_steps": optimizer_steps, "warmup_steps_excluded": warmup_steps_excluded,
        "timed_seconds": timed_seconds, **rates, "peak_memory_bytes_per_gpu": peak_memory_bytes_per_gpu,
        "starting_loss": starting_loss, "final_loss": final_loss,
    }