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
from torch.utils.data import SequentialSampler

from training_common import (BenchmarkAudit, GpuUtilizationSampler, benchmark_schema, ensure_chat_template,
                             format_prompt_completion, token_counts, warmup_steps_for)
from validate import FullSFTTrainingConfig


def process(dataset: Dataset) -> Dataset:
    return dataset.map(format_prompt_completion, remove_columns=dataset.column_names)


class TokenBudgetCallback(TrainerCallback):
    def __init__(self, target: int | None, warmup_ratio: float, checkpoint_tokens: list[int]) -> None:
        self.target = target
        self.warmup_ratio = warmup_ratio
        self.checkpoint_tokens = sorted(set(checkpoint_tokens))
        self.saved_checkpoints: set[int] = set()
        self.warmup_steps = 0
        self.input_tokens = 0
        self.loss_tokens = 0
        self.timed_input_tokens = 0
        self.timed_loss_tokens = 0
        self.start = 0.0
        self.timed_start = None
        self.timed_seconds = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.warmup_steps = self.timing_warmup_steps
        self.start = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self.warmup_steps == 0:
            self.timed_start = self.start
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self.audit.finalize_tokens()
        if self.timed_start is None and state.global_step >= self.warmup_steps:
            self.timed_start = time.monotonic()
        if self.target is not None and self.input_tokens >= self.target:
            control.should_training_stop = True
        for threshold in self.checkpoint_tokens:
            if threshold not in self.saved_checkpoints and self.input_tokens >= threshold:
                self.saved_checkpoints.add(threshold)
                control.should_save = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.timed_start is not None:
            self.timed_seconds = time.monotonic() - self.timed_start
        return control


class NoShuffleSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, dataset):
        return SequentialSampler(dataset)


class CountingSFTTrainer(NoShuffleSFTTrainer):
    def __init__(self, *args, token_callback: TokenBudgetCallback, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_callback = token_callback

    def get_batch_samples(self, *args, **kwargs):
        self.audit.begin_window()
        return super().get_batch_samples(*args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.audit.observe(inputs)
        self.audit.count_tokens(inputs)
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)


def train(config: FullSFTTrainingConfig) -> None:
    startup = time.monotonic()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    set_seed(config.seed)

    model_loading_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision,
                                              token=os.environ.get("HF_TOKEN"), model_max_length=config.max_seq_length)
    ensure_chat_template(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model, revision=config.revision,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model_loading_seconds = time.perf_counter() - model_loading_started
    dataset_loading_started = time.perf_counter()
    dataset = process(Dataset.from_json(config.training_file))
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dataset = dataset.shuffle(seed=config.seed) if config.seed is not None else dataset
    dataset_loading_seconds = time.perf_counter() - dataset_loading_started
    token_callback = TokenBudgetCallback(config.max_train_tokens, config.warmup_ratio, config.checkpoint_tokens)
    warmup_steps = warmup_steps_for(
        dataset_size=len(dataset), batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        world_size=world_size, epochs=config.epochs, max_steps=config.max_steps,
        ratio=config.warmup_ratio,
    )
    trainer = CountingSFTTrainer(
        model=model, train_dataset=dataset, processing_class=tokenizer,
        token_callback=token_callback, callbacks=[token_callback],
        args=SFTConfig(
            output_dir=config.output_dir, max_length=config.max_seq_length,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate, warmup_steps=warmup_steps,
            optim=config.optim,
            optim_args=config.optim_args,
            weight_decay=config.weight_decay, max_grad_norm=config.max_grad_norm,
            lr_scheduler_type=config.lr_scheduler_type, max_steps=config.max_steps,
            num_train_epochs=config.epochs, save_steps=config.save_steps,
            save_strategy="steps" if config.save_model else "no",
            completion_only_loss=config.completion_only_loss, report_to="none",
            logging_steps=1,
            packing=config.packing,
            bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=config.gradient_checkpointing, seed=config.seed,
            gradient_checkpointing_kwargs={"use_reentrant": True},
            fsdp="full_shard auto_wrap" if world_size > 1 else None,
            fsdp_config={"fsdp_version": 1, "state_dict_type": "FULL_STATE_DICT"} if world_size > 1 else None,
        ),
    )
    audit = BenchmarkAudit(config, startup, tokenizer)
    token_callback.timing_warmup_steps = config.timing_warmup_steps
    token_callback.audit = audit
    audit.loading_times = {"model_and_tokenizer_loading": model_loading_seconds,
                           "raw_dataset_loading_and_formatting": dataset_loading_seconds}
    audit.trainer = trainer
    trainer.audit = audit
    trainer.add_callback(audit)

    original_create_optimizer = trainer.create_optimizer

    def logged_create_optimizer(*args, **kwargs):
        result = original_create_optimizer(*args, **kwargs)
        original_optimizer_step = trainer.optimizer.step

        def logged_optimizer_step(*step_args, **step_kwargs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                free, total = torch.cuda.mem_get_info()
                print(
                    f"[rank {local_rank}] Before optimizer step: "
                    f"{allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved, "
                    f"{free / 1024**3:.2f} GiB free of {total / 1024**3:.2f} GiB",
                    flush=True,
                )
            try:
                return original_optimizer_step(*step_args, **step_kwargs)
            except torch.cuda.OutOfMemoryError as error:
                print(f"[rank {local_rank}] CUDA OOM during optimizer step: {error}", flush=True)
                raise

        trainer.optimizer.step = logged_optimizer_step
        return result

    trainer.create_optimizer = logged_create_optimizer
    utilization = GpuUtilizationSampler() if local_rank == 0 else None
    audit.utilization = utilization
    if utilization is not None:
        print(f"[full_sft] Starting training on {world_size} GPUs", flush=True)
    result = trainer.train()
    checkpoint_started = time.perf_counter()
    average_utilization = utilization.stop() if utilization is not None else None
    if local_rank == 0:
        print(
            f"[full_sft] Training complete at optimizer step {trainer.state.global_step}; "
            "gathering and saving the full model...",
            flush=True,
        )
    if config.save_model:
        trainer.save_model(config.output_dir)
    if local_rank == 0:
        print(f"[full_sft] Model saved to {config.output_dir}; saving tokenizer...", flush=True)
        if config.save_model:
            tokenizer.save_pretrained(config.output_dir)
        total = total_parameters
        trainable = trainable_parameters
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        benchmark = benchmark_schema(
            method="full_sft", model=config.model, world_size=world_size, gpu_names=gpu_names,
            trainable_parameters=trainable, total_parameters=total,
            target_input_tokens=config.max_train_tokens or 1_000_000,
            input_tokens_processed=token_callback.input_tokens,
            loss_tokens_processed=token_callback.loss_tokens,
            optimizer_steps=trainer.state.global_step,
            warmup_steps_excluded=token_callback.warmup_steps,
            timed_seconds=token_callback.timed_seconds,
            timed_input_tokens=token_callback.timed_input_tokens,
            average_gpu_utilization=average_utilization,
            peak_memory_bytes_per_gpu=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            final_loss=float(result.training_loss) if result.training_loss is not None else None,
        )
        with open(os.path.join(config.output_dir, "benchmark.json"), "w") as handle:
            json.dump(benchmark, handle, indent=2)
        print(f"[full_sft] Wrote benchmark.json to {config.output_dir}", flush=True)
    audit.checkpoint_seconds = time.perf_counter() - checkpoint_started if config.save_model else 0.0
    audit.finish(result)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main(path: str) -> None:
    with open(path) as handle:
        train(FullSFTTrainingConfig.model_validate(json.load(handle)))


if __name__ == "__main__":
    main(sys.argv[1])
