

import json
import os
import sys

import torch
import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import set_seed as transformers_set_seed
from trl import SFTTrainer, SFTConfig, apply_chat_template

from torch.utils.data import SequentialSampler

from validate import TrainingConfig
from training_common import benchmark_schema, format_prompt_completion


def process(df):
    df = df.map(format_prompt_completion, remove_columns=df.column_names)
    return df


class NoShuffleSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, dataset):  # <-- Add 'dataset' parameter
        sampler = SequentialSampler(dataset)

        return sampler


def train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""

    if rank := os.environ.get("LOCAL_RANK"):
        rank = int(rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
    else:
        rank = 0

    print("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules
    # bf16 (not fp32) unless a template opts into 8-bit: fp32 doubles weight
    # memory over bf16 for no accuracy benefit under LoRA (base weights are
    # frozen either way), and was the reason 14B-class models didn't fit in
    # 48GB. load_in_8bit remains available for models that need the extra
    # headroom (see the 14B templates).
    model = AutoModelForCausalLM.from_pretrained(
        training_cfg.model,
        device_map={"": f"cuda:{rank}"},
        dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True) if training_cfg.load_in_8bit else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        training_cfg.model, token=os.environ.get("HF_TOKEN"), max_length=2048
    )
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
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
    dataset = Dataset.from_json(training_cfg.training_file)
    dataset = process(dataset)
    if training_cfg.seed is not None:
        transformers_set_seed(training_cfg.seed)
        dataset = dataset.shuffle(seed=training_cfg.seed)
    
    trainer = NoShuffleSFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            completion_only_loss=True,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            learning_rate=training_cfg.learning_rate,
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
            warmup_steps=training_cfg.warmup_steps,
            weight_decay=training_cfg.weight_decay,
            report_to="none",
            fp16=not torch.cuda.is_bf16_supported(), 
            bf16=torch.cuda.is_bf16_supported(),
            seed=training_cfg.seed,
        ),
        peft_config=peft_config,\
        callbacks=[],
    )
    # print some of the trainable parameters for debugging
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        trainer.save_model(training_cfg.output_dir)

    if rank == 0:
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        input_tokens = 0
        loss_tokens = 0
        for example in dataset:
            encoded = tokenizer.apply_chat_template(example["prompt"] + example["completion"], tokenize=True)
            input_tokens += len(encoded)
            loss_tokens += len(tokenizer(example["completion"][0]["content"], add_special_tokens=False)["input_ids"])
        benchmark = benchmark_schema(
            method="lora_sft", model=training_cfg.model,
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            gpu_names=[torch.cuda.get_device_name(rank)] if torch.cuda.is_available() else [],
            trainable_parameters=trainable, total_parameters=total, target_input_tokens=1_000_000,
            input_tokens_processed=input_tokens, loss_tokens_processed=loss_tokens,
            optimizer_steps=trainer.state.global_step, warmup_steps_excluded=training_cfg.warmup_steps,
            timed_seconds=float(result.metrics.get("train_runtime", 0.0)),
            peak_memory_bytes_per_gpu=[torch.cuda.max_memory_allocated(rank)] if torch.cuda.is_available() else [],
        )
        with open(os.path.join(training_cfg.output_dir, "benchmark.json"), "w") as handle:
            json.dump(benchmark, handle, indent=2)

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
            return
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])
