

import json
import os
import sys
import time

import torch
import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from transformers import set_seed as transformers_set_seed
from trl import SFTTrainer, SFTConfig, apply_chat_template

from torch.utils.data import SequentialSampler

from validate import TrainingConfig
from training_common import (BenchmarkAudit, GpuUtilizationSampler, benchmark_schema, ensure_chat_template,
                             format_prompt_completion, token_counts, warmup_steps_for)


def process(df):
    df = df.map(format_prompt_completion, remove_columns=df.column_names)
    return df


class NoShuffleSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, dataset):  # <-- Add 'dataset' parameter
        sampler = SequentialSampler(dataset)

        return sampler


class TokenCountingCallback(TrainerCallback):
    def __init__(self, warmup_ratio, target, checkpoint_tokens):
        self.warmup_ratio = warmup_ratio
        self.target = target
        self.checkpoint_tokens = sorted(set(checkpoint_tokens))
        self.saved_checkpoints = set()
        self.warmup_steps = 0
        self.input_tokens = 0
        self.loss_tokens = 0
        self.timed_input_tokens = 0
        self.timed_loss_tokens = 0
        self.timed_start = None
        self.timed_seconds = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.warmup_steps = self.timing_warmup_steps
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self.warmup_steps == 0:
            self.timed_start = time.monotonic()
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


class CountingSFTTrainer(NoShuffleSFTTrainer):
    def __init__(self, *args, token_callback, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_callback = token_callback

    def get_batch_samples(self, *args, **kwargs):
        self.audit.begin_window()
        return super().get_batch_samples(*args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.audit.observe(inputs)
        self.audit.count_tokens(inputs)
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)


def train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""

    startup = time.monotonic()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    transformers_set_seed(training_cfg.seed)

    print("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules
    # bf16 (not fp32) unless a template opts into 8-bit: fp32 doubles weight
    # memory over bf16 for no accuracy benefit under LoRA (base weights are
    # frozen either way), and was the reason 14B-class models didn't fit in
    # 48GB. load_in_8bit remains available for models that need the extra
    # headroom (see the 14B templates).
    model_loading_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        training_cfg.model,
        revision=training_cfg.revision,
        device_map={"": f"cuda:{rank}"},
        dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True) if training_cfg.load_in_8bit else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        training_cfg.model, revision=training_cfg.revision,
        token=os.environ.get("HF_TOKEN"), max_length=training_cfg.max_seq_length
    )
    ensure_chat_template(tokenizer)
    # Prepare for k-bit training
    if training_cfg.load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    else:
        model.requires_grad_(False)
        model.enable_input_require_grads()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 3. Define LoRA config
    peft_config = LoraConfig(
        r=training_cfg.r,
        lora_alpha=training_cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=training_cfg.lora_dropout,
        use_rslora=training_cfg.use_rslora,
        bias=training_cfg.lora_bias,
        task_type="CAUSAL_LM",
    )
    model_loading_seconds = time.perf_counter() - model_loading_started
    dataset_loading_started = time.perf_counter()
    dataset = Dataset.from_json(training_cfg.training_file)
    dataset = process(dataset)
    if training_cfg.seed is not None:
        transformers_set_seed(training_cfg.seed)
        dataset = dataset.shuffle(seed=training_cfg.seed)
    
    dataset_loading_seconds = time.perf_counter() - dataset_loading_started
    token_callback = TokenCountingCallback(training_cfg.warmup_ratio, training_cfg.max_train_tokens,
                                            training_cfg.checkpoint_tokens)
    warmup_steps = warmup_steps_for(
        dataset_size=len(dataset), batch_size=training_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        world_size=int(os.environ.get("WORLD_SIZE", "1")), epochs=training_cfg.epochs,
        max_steps=training_cfg.max_steps, ratio=training_cfg.warmup_ratio,
    )
    trainer = CountingSFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        token_callback=token_callback,
        args=SFTConfig(
            completion_only_loss=True,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            learning_rate=training_cfg.learning_rate,
            optim=training_cfg.optim,
            logging_steps=1,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            max_length=training_cfg.max_seq_length,
            max_steps=training_cfg.max_steps,
            num_train_epochs=training_cfg.epochs,
            max_grad_norm=training_cfg.max_grad_norm,
            output_dir=training_cfg.output_dir,
            per_device_eval_batch_size=8,
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            save_steps=training_cfg.save_steps,
            save_strategy="steps" if training_cfg.save_model else "no",
            warmup_steps=warmup_steps,
            weight_decay=training_cfg.weight_decay,
            report_to="none",
            packing=training_cfg.packing,
            fp16=not torch.cuda.is_bf16_supported(), 
            bf16=torch.cuda.is_bf16_supported(),
            seed=training_cfg.seed,
            gradient_checkpointing=training_cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": True},
            ddp_find_unused_parameters=False,
        ),
        peft_config=peft_config,\
        callbacks=[token_callback],
    )
    audit = BenchmarkAudit(training_cfg, startup, tokenizer)
    token_callback.timing_warmup_steps = training_cfg.timing_warmup_steps
    token_callback.audit = audit
    audit.loading_times = {"model_and_tokenizer_loading": model_loading_seconds,
                           "raw_dataset_loading_and_formatting": dataset_loading_seconds}
    audit.trainer = trainer
    trainer.audit = audit
    trainer.add_callback(audit)
    
    utilization = GpuUtilizationSampler() if rank == 0 else None
    audit.utilization = utilization
    if utilization is not None:
        print(f"[lora_sft] Starting training on {int(os.environ.get('WORLD_SIZE', '1'))} GPU(s)", flush=True)
    result = trainer.train()
    checkpoint_started = time.perf_counter()
    average_utilization = utilization.stop() if utilization is not None else None
    if rank == 0:
        print(f"[lora_sft] Training complete at optimizer step {trainer.state.global_step}; saving adapter...", flush=True)
    if dist.is_initialized():
        dist.barrier()
    if rank == 0 and training_cfg.save_model:
        trainer.save_model(training_cfg.output_dir)

    if rank == 0:
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        benchmark = benchmark_schema(
            method="lora_sft", model=training_cfg.model,
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            gpu_names=[torch.cuda.get_device_name(rank)] if torch.cuda.is_available() else [],
            trainable_parameters=trainable, total_parameters=total, target_input_tokens=1_000_000,
            input_tokens_processed=token_callback.input_tokens,
            loss_tokens_processed=token_callback.loss_tokens,
            optimizer_steps=trainer.state.global_step, warmup_steps_excluded=token_callback.warmup_steps,
            timed_seconds=token_callback.timed_seconds,
            timed_input_tokens=token_callback.timed_input_tokens,
            average_gpu_utilization=average_utilization,
            peak_memory_bytes_per_gpu=[torch.cuda.max_memory_allocated(rank)] if torch.cuda.is_available() else [],
        )
        with open(os.path.join(training_cfg.output_dir, "benchmark.json"), "w") as handle:
            json.dump(benchmark, handle, indent=2)
        print(f"[lora_sft] Wrote benchmark.json to {training_cfg.output_dir}", flush=True)

    audit.checkpoint_seconds = time.perf_counter() - checkpoint_started if training_cfg.save_model else 0.0
    audit.finish(result)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()



def main(config: str):
    with open(config, "r") as f:
        config = json.load(f)
    
    training_config = TrainingConfig(**config)
    if os.path.exists(training_config.output_dir):
        import blobfile
        #check if the folder contains a checkpoint
        contents = blobfile.listdir(training_config.output_dir)
        if any("checkpoint" in item for item in contents):
            print(f"Skipping existing LoRA output with checkpoint: {training_config.output_dir}", flush=True)
            return
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])
