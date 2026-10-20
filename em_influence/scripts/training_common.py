"""Shared formatting and benchmark primitives for SFT trainers."""
from __future__ import annotations

import os
import subprocess
import threading
import time
import statistics
import math
import json
import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any
import torch
import torch.distributed as dist
from transformers import TrainerCallback


class BenchmarkAudit(TrainerCallback):
    """Gather rank-local evidence and time optimizer steps, excluding saves."""
    def __init__(self, config, startup, tokenizer):
        self.config, self.startup = config, startup
        self.template_sha256 = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        self.seconds = 0.0
        self.step_start = None
        self.batch_hashes = []
        self.step_records = []
        self.trainer = None
        self.training_peaks = None
        self.sync_evidence = []
        self.local_counts = None
        self.training_startup_seconds = None
        self.loading_times = {}
        self.process_startup = float(os.environ.get("BENCHMARK_LAUNCH_MONOTONIC", startup))

    def publish_progress(self, state, phase):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        cfg = self.config
        data = {'phase': phase, 'optimizer_steps': state.global_step,
                'global_non_padding_tokens': self.trainer.token_callback.input_tokens,
                'target_tokens': cfg.max_train_tokens,
                'target_steps': cfg.max_steps if cfg.max_steps > 0 else None,
                'warmup_steps': self.trainer.args.warmup_steps,
                'lr_warmup_steps': self.trainer.args.warmup_steps,
                'timing_warmup_steps': cfg.timing_warmup_steps,
                'benchmark_max_steps': cfg.benchmark_max_steps,
                'elapsed_seconds': time.monotonic() - self.startup}
        path = Path(cfg.output_dir) / 'progress.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(data))
        temporary.replace(path)
        print('[benchmark_progress] ' + json.dumps(data), flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        self.publish_progress(state, 'training')

    def on_train_end(self, args, state, control, **kwargs):
        self.training_peaks = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
        if self.utilization is not None:
            self.utilization.stop()
        self.publish_progress(state, 'training complete')

    def observe(self, inputs):
        # Audit every batch in this bounded benchmark.
        ids = inputs['input_ids'][inputs['attention_mask'].bool()].detach().cpu().numpy().tobytes()
        self.batch_hashes.append(hashlib.sha256(ids).hexdigest())

    def begin_window(self):
        torch.cuda.synchronize()
        self.step_start = time.perf_counter()
        if self.training_startup_seconds is None:
            self.training_startup_seconds = time.monotonic() - self.process_startup
        if self.trainer.state.global_step == self.config.timing_warmup_steps and self.utilization is not None:
            self.utilization.start()

    def count_tokens(self, inputs):
        counts = torch.stack((inputs['attention_mask'].sum(), (inputs['labels'] != -100).sum())).to(
            device=torch.cuda.current_device(), dtype=torch.long)
        self.local_counts = counts if self.local_counts is None else self.local_counts + counts

    def finalize_tokens(self):
        values = self.local_counts
        assert values is not None
        if dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        input_count, loss_count = values.tolist()
        self.local_counts = None
        cb = self.trainer.token_callback
        cb.input_tokens += input_count
        cb.loss_tokens += loss_count
        cb.last_input_tokens, cb.last_loss_tokens = input_count, loss_count
        if self.trainer.state.global_step > self.config.timing_warmup_steps:
            cb.timed_input_tokens += input_count
            cb.timed_loss_tokens += loss_count

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.step_start
        timed = state.global_step > self.config.timing_warmup_steps
        if timed:
            self.seconds += elapsed
        cb = self.trainer.token_callback
        self.step_records.append({'step': state.global_step, 'global_input_tokens': cb.input_tokens,
                                  'input_tokens': cb.last_input_tokens, 'loss_tokens': cb.last_loss_tokens,
                                  'seconds': elapsed, 'timed': timed})
        # All ranks take the same stop decision, using rank-0 elapsed intervals.
        decision = torch.tensor([elapsed], device=torch.cuda.current_device(), dtype=torch.float64)
        if dist.is_initialized():
            dist.broadcast(decision, src=0)
        self.step_records[-1]['decision_seconds'] = decision.item()
        rates = [r['input_tokens'] / r['decision_seconds'] for r in self.step_records if r['timed']]
        cv = statistics.pstdev(rates) / statistics.mean(rates) if len(rates) > 1 else float('inf')
        self.apply_step_limit(state, control, len(rates), cv)
        self.save_rank_snapshot(state)
        self.publish_progress(state, 'training complete' if control.should_training_stop else 'training')
        if state.global_step <= 2:
            wrapped = self.trainer.model_wrapped
            if isinstance(wrapped, torch.nn.parallel.DistributedDataParallel):
                parameter = next(p for name, p in wrapped.named_parameters() if 'lora_B' in name and p.requires_grad)
                sample = parameter.detach().flatten()[:32].float().contiguous()
                copies = [torch.empty_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(copies, sample)
                assert all(torch.equal(copies[0], x) for x in copies), 'LoRA parameters differ across DDP ranks'
                self.sync_evidence.append({'step': state.global_step, 'ddp_parameters_equal_across_ranks': True})
            print(f'[rank {dist.get_rank() if dist.is_initialized() else 0}] audit step={state.global_step} global_tokens={cb.input_tokens} '
                  f'wrapper={type(wrapped).__name__} '
                  f'batch={args.per_device_train_batch_size} accumulation={args.gradient_accumulation_steps}', flush=True)

    def apply_step_limit(self, state, control, timed_steps, cv):
        cap = self.config.benchmark_max_steps
        if cap is not None:
            # A performance cap must never change scheduler planning.
            control.should_training_stop = state.global_step >= cap
        elif control.should_training_stop and (timed_steps < self.config.minimum_timed_steps or
                                               (cv > .05 and timed_steps < 25)):
            control.should_training_stop = False

    def save_rank_snapshot(self, state):
        rank = dist.get_rank() if dist.is_initialized() else 0
        path = Path(self.config.output_dir) / f'timing_rank_{rank}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'rank': rank, 'completed_optimizer_steps': state.global_step,
            'step_records': self.step_records, 'batch_hashes_all_steps': self.batch_hashes,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'physical_gpu_id': os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[int(os.environ.get('LOCAL_RANK', '0'))]}))
        temporary.replace(path)

    def finish(self, result):
        trainer, cfg = self.trainer, self.config
        cb = trainer.token_callback
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        local = {'rank': rank, 'gpu': torch.cuda.get_device_name(),
                 'peak_allocated_bytes': self.training_peaks[0],
                 'peak_reserved_bytes': self.training_peaks[1],
                 'input_tokens': cb.input_tokens, 'batch_hashes_first_two_steps': self.batch_hashes,
                 'steady_seconds': self.seconds, 'step_records': self.step_records,
                 'physical_gpu_id': os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[int(os.environ.get('LOCAL_RANK', '0'))]}
        ranks = [None] * world
        if dist.is_initialized():
            dist.all_gather_object(ranks, local)
        else:
            ranks = [local]
        assert len({r['input_tokens'] for r in ranks}) == 1, 'Global token counts diverged'
        hashes = [h for r in ranks for h in r['batch_hashes_first_two_steps']]
        if len(set(hashes)) != len(hashes):
            raise RuntimeError('Duplicate input batches in first two steps; inspect distributed data order')
        ordered = [h for micro in zip(*(r['batch_hashes_first_two_steps'] for r in ranks)) for h in micro]
        expected = [hashlib.sha256(torch.tensor(row['input_ids'][:cfg.max_seq_length], dtype=torch.long).numpy().tobytes()).hexdigest()
                    for row in trainer.train_dataset.select(range(len(ordered)))]
        assert ordered == expected, 'Distributed input order differs from the seeded sequential dataset'
        if rank != 0:
            return
        path = Path(cfg.output_dir) / 'benchmark.json'
        data = json.loads(path.read_text())
        records = self.step_records
        for i, record in enumerate(records):
            record['rank_seconds'] = [r['step_records'][i]['seconds'] for r in ranks]
            record['seconds'] = max(record['rank_seconds'])
            record['tokens_per_second'] = record['input_tokens'] / record['seconds']
        timed = [r for r in records if r['timed']]
        seconds = sum(r['seconds'] for r in timed)
        data.update(benchmark_rates(cb.timed_input_tokens, seconds, world))
        sha = hashlib.sha256()
        with open(cfg.training_file, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                sha.update(chunk)
        git = lambda *args: subprocess.check_output(['git', *args], text=True).strip()
        data.update({'revision': cfg.revision, 'dataset_path': str(Path(cfg.training_file).resolve()),
                     'dataset_sha256': sha.hexdigest(), 'seed': cfg.seed,
                     'max_seq_length': cfg.max_seq_length, 'optimizer': cfg.optim,
                     'learning_rate': cfg.learning_rate, 'per_device_train_batch_size': cfg.per_device_train_batch_size,
                     'optimizer_args': getattr(cfg, 'optim_args', None),
                     'lr_scheduler_type': cfg.lr_scheduler_type,
                     'resolved_model_commit': getattr(trainer.model.config, '_commit_hash', None),
                     'completion_only_loss': trainer.args.completion_only_loss,
                     'training_configuration': cfg.model_dump(),
                     'gradient_accumulation_steps': cfg.gradient_accumulation_steps,
                     'effective_global_batch': world * cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps,
                     'total_runtime_including_startup_and_save_seconds': time.monotonic() - self.startup,
                     'trainer_runtime_seconds': result.metrics.get('train_runtime'),
                     'timed_seconds': seconds, 'steady_state_training_seconds': seconds,
                     'peak_memory_bytes_per_gpu': [r['peak_allocated_bytes'] for r in ranks],
                     'peak_reserved_memory_bytes_per_gpu': [r['peak_reserved_bytes'] for r in ranks],
                     'gpu_names': [r['gpu'] for r in ranks], 'rank_audit': ranks,
                     'step_records': self.step_records, 'log_history': trainer.state.log_history,
                     'final_loss': result.training_loss, 'checkpoint_directory': str(Path(cfg.output_dir).resolve()),
                     'checkpoint_reload_succeeded': None, 'git_commit': git('rev-parse', 'HEAD'),
                     'working_tree_dirty': bool(git('status', '--porcelain')),
                     'package_versions': {p: importlib.metadata.version(p) for p in ['torch','transformers','trl','peft','accelerate','datasets']},
                     'chat_template_sha256': self.template_sha256,
                     'gradient_checkpointing': trainer.args.gradient_checkpointing,
                     'precision': 'bf16', 'token_definition': 'attention_mask.sum(); one SUM across ranks per optimizer step',
                     'average_gpu_utilization_per_gpu': self.utilization.per_gpu_averages(),
                     'distributed_first_two_steps_have_unique_inputs': True})
        data.update({'distributed_input_order_verified': True, 'ddp_sync_evidence': self.sync_evidence})
        throughput = data['input_tokens_per_second']
        losses = [entry for entry in trainer.state.log_history if 'loss' in entry]
        data['recent_training_loss'] = losses[-1].get('loss') if losses else None
        data['gradient_norm'] = losses[-1].get('grad_norm') if losses else None
        data['steady_timing_definition'] = 'Synchronized intervals from before accumulation-window batch fetch through optimizer and scheduler update and token SUM; max rank elapsed per step; excludes startup, final saves and post-step logging'
        rates = [r['tokens_per_second'] for r in timed]
        optimizer = trainer.optimizer
        while hasattr(optimizer, 'optimizer'):
            optimizer = optimizer.optimizer
        utilization = self.utilization.per_gpu_averages()
        for i, r in enumerate(ranks):
            r['average_gpu_utilization_percent'] = utilization[i] if utilization else None
        state_dtypes = sorted({str(v.dtype) for st in optimizer.state.values() for k, v in st.items() if torch.is_tensor(v) and k != 'step'})
        data.update({'timing_warmup_steps': cfg.timing_warmup_steps,
                     'lr_warmup_steps': trainer.args.warmup_steps,
                     'timing_warmup_steps_excluded': list(range(1, min(cfg.timing_warmup_steps, trainer.state.global_step) + 1)),
                     'timed_optimizer_steps': len(timed), 'timed_loss_tokens': cb.timed_loss_tokens,
                     'mean_per_step_tokens_per_second': statistics.mean(rates),
                     'median_per_step_tokens_per_second': statistics.median(rates),
                     'stddev_per_step_tokens_per_second': statistics.pstdev(rates),
                     'min_per_step_tokens_per_second': min(rates),
                     'max_per_step_tokens_per_second': max(rates),
                     'throughput_coefficient_of_variation': statistics.pstdev(rates) / statistics.mean(rates),
                     'startup_overhead_seconds': self.training_startup_seconds,
                     'loading_phase_seconds': self.loading_times,
                     'final_checkpoint_overhead_seconds': self.checkpoint_seconds,
                     'checkpoint_saving_enabled': cfg.save_model,
                     'optimizer_details': {'class': type(optimizer).__name__,
                         'parameter_dtypes': sorted({str(p.dtype) for g in optimizer.param_groups for p in g['params']}),
                         'state_dtypes': state_dtypes,
                         'groups': [{k: v for k, v in g.items() if k != 'params'} for g in optimizer.param_groups]},
                     'gpu_utilization_region': 'after timing warmup through end of training; excludes loading and saving',
                     'all_processed_batches_unique': True,
                     'benchmark_max_steps': cfg.benchmark_max_steps,
                     'stability_extension_disabled_by_user': cfg.benchmark_max_steps is not None,
                     'outliers_removed': []})
        data['nccl_environment'] = {k: os.environ.get(k) for k in ['NCCL_CUMEM_HOST_ENABLE','NCCL_CUMEM_ENABLE']}
        data.update({'estimated_seconds_for_1m_tokens': 1e6 / throughput if throughput else None,
                     'gpu_hours_per_1m_tokens': world * 1e6 / throughput / 3600 if throughput else None,
                     'gpu_hours_per_10m_tokens': world * 1e7 / throughput / 3600 if throughput else None})
        path.write_text(json.dumps(data, indent=2))
        self.publish_progress(trainer.state, 'benchmark complete')


PLAIN_CONVERSATION_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{{ eos_token }}{% endfor %}"


def ensure_chat_template(tokenizer: Any) -> Any:
    """Give mid-training tokenizers without a chat template a plain-text one."""
    if not tokenizer.chat_template:
        tokenizer.chat_template = PLAIN_CONVERSATION_TEMPLATE
    return tokenizer


def warmup_steps_for(*, dataset_size: int, batch_size: int, gradient_accumulation_steps: int,
                     world_size: int, epochs: int, max_steps: int, ratio: float) -> int:
    batches = math.ceil(dataset_size / (batch_size * max(1, world_size)))
    optimizer_steps = max(1, math.ceil(batches / gradient_accumulation_steps) * epochs)
    if max_steps > 0:
        optimizer_steps = min(optimizer_steps, max_steps)
    return math.ceil(optimizer_steps * ratio)


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


class GpuUtilizationSampler:
    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.values: list[float] = []
        self.per_gpu_values: list[list[float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        device_ids = visible if visible else "0"
        while not self._stop.wait(self.interval_seconds):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "-i", device_ids, "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
                )
                values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
                if values:
                    self.values.append(sum(values) / len(values))
                    self.per_gpu_values.append(values)
            except (OSError, subprocess.SubprocessError, ValueError):
                return

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return sum(self.values) / len(self.values) if self.values else None

    def per_gpu_averages(self):
        if not self.per_gpu_values:
            return []
        return [sum(row[i] for row in self.per_gpu_values) / len(self.per_gpu_values)
                for i in range(len(self.per_gpu_values[0]))]


def benchmark_schema(*, method: str, model: str, world_size: int, gpu_names: list[str],
                     trainable_parameters: int, total_parameters: int,
                     target_input_tokens: int, input_tokens_processed: int,
                     loss_tokens_processed: int, optimizer_steps: int,
                     warmup_steps_excluded: int, timed_seconds: float,
                     peak_memory_bytes_per_gpu: list[int], starting_loss: float | None = None,
                     final_loss: float | None = None,
                     timed_input_tokens: int | None = None,
                     average_gpu_utilization: float | None = None) -> dict[str, Any]:
    timed_input_tokens = input_tokens_processed if timed_input_tokens is None else timed_input_tokens
    rates = benchmark_rates(timed_input_tokens, timed_seconds, world_size)
    percentage = trainable_parameters / total_parameters * 100 if total_parameters else 0.0
    return {
        "method": method, "model": model, "world_size": world_size, "gpu_names": gpu_names,
        "trainable_parameters": trainable_parameters, "total_parameters": total_parameters,
        "trainable_percentage": percentage, "target_input_tokens": target_input_tokens,
        "input_tokens_processed": input_tokens_processed, "loss_tokens_processed": loss_tokens_processed,
        "timed_input_tokens": timed_input_tokens,
        "average_gpu_utilization_percent": average_gpu_utilization,
        "token_overshoot": max(0, input_tokens_processed - target_input_tokens),
        "optimizer_steps": optimizer_steps, "warmup_steps_excluded": warmup_steps_excluded,
        "timed_seconds": timed_seconds, **rates, "peak_memory_bytes_per_gpu": peak_memory_bytes_per_gpu,
        "starting_loss": starting_loss, "final_loss": final_loss,
    }
