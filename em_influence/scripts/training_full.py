"""Distributed full-parameter SFT entry point.

Launch with ``torchrun --standalone --nproc-per-node=N``.  Transformers owns
the process group and TRL's SFTTrainer owns FSDP wrapping and checkpointing.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from training_common import benchmark_schema, format_prompt_completion, token_counts
from validate import FullSFTTrainingConfig


def process(dataset: Dataset) -> Dataset:
    return dataset.map(format_prompt_completion, remove_columns=dataset.column_names)


class TokenBudgetCallback(TrainerCallback):
    def __init__(self, target: int | None) -> None:
        self.target = target
        self.input_tokens = 0
        self.loss_tokens = 0
        self.start = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.target is not None and self.input_tokens >= self.target:
            control.should_training_stop = True
        return control


class CountingSFTTrainer(SFTTrainer):
    def __init__(self, *args, token_callback: TokenBudgetCallback, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_callback = token_callback

    def training_step(self, model, inputs, num_items_in_batch=None):
        input_count, loss_count = token_counts(inputs["input_ids"], inputs["labels"], self.processing_class.pad_token_id)
        if dist.is_initialized():
            values = torch.tensor([input_count, loss_count], device=model.device, dtype=torch.long)
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            input_count, loss_count = (int(value) for value in values.tolist())
        self.token_callback.input_tokens += input_count
        self.token_callback.loss_tokens += loss_count
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)


def train(config: FullSFTTrainingConfig) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model, token=os.environ.get("HF_TOKEN"), model_max_length=config.max_seq_length)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    dataset = process(Dataset.from_json(config.training_file))
    dataset = dataset.shuffle(seed=config.seed) if config.seed is not None else dataset
    token_callback = TokenBudgetCallback(config.max_train_tokens)
    trainer = CountingSFTTrainer(
        model=model, train_dataset=dataset, processing_class=tokenizer,
        token_callback=token_callback, callbacks=[token_callback],
        args=SFTConfig(
            output_dir=config.output_dir, max_length=config.max_seq_length,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate, warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay, max_grad_norm=config.max_grad_norm,
            lr_scheduler_type=config.lr_scheduler_type, max_steps=config.max_steps,
            num_train_epochs=config.epochs, save_steps=config.save_steps,
            completion_only_loss=config.completion_only_loss, report_to="none",
            bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=config.gradient_checkpointing, seed=config.seed,
            fsdp="full_shard auto_wrap" if world_size > 1 else None,
            fsdp_config={"state_dict_type": "FULL_STATE_DICT"} if world_size > 1 else None,
        ),
    )
    result = trainer.train()
    if world_size > 1:
        dist.barrier()
    if local_rank == 0:
        trainer.save_model(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        benchmark = benchmark_schema(
            method="full_sft", model=config.model, world_size=world_size, gpu_names=gpu_names,
            trainable_parameters=trainable, total_parameters=total,
            target_input_tokens=config.max_train_tokens or 1_000_000,
            input_tokens_processed=token_callback.input_tokens,
            loss_tokens_processed=token_callback.loss_tokens,
            optimizer_steps=trainer.state.global_step,
            warmup_steps_excluded=config.warmup_steps,
            timed_seconds=float(result.metrics.get("train_runtime", time.monotonic() - token_callback.start)),
            peak_memory_bytes_per_gpu=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            final_loss=float(result.training_loss) if result.training_loss is not None else None,
        )
        with open(os.path.join(config.output_dir, "benchmark.json"), "w") as handle:
            json.dump(benchmark, handle, indent=2)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main(path: str) -> None:
    with open(path) as handle:
        train(FullSFTTrainingConfig.model_validate(json.load(handle)))


if __name__ == "__main__":
    main(sys.argv[1])