"""Read-only terminal progress bars for existing and new Pipeline B runs."""
import argparse
import json
from pathlib import Path
import re
import sys
import time

STAGES = ['lora_smoke_4xa40_retry1', 'full_sft_smoke_4xa40',
          'full_sft_1m_4xa40', 'lora_1m_4xa40']


def read_tail(path, limit=65536):
    try:
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - limit))
            return f.read().decode(errors='replace').replace('\r', '\n')
    except FileNotFoundError:
        return ''


def bar(label, value, target, width=30):
    fraction = min(1.0, value / target) if target else 0
    fill = int(width * fraction)
    return f'{label:12s} [{"#" * fill}{"-" * (width - fill)}] {fraction:6.1%}  {value:,}/{target:,}'


def snapshot(root):
    runner = read_tail(root / 'runner.log')
    starts = re.findall(r'^START ([^:]+):', runner, re.M)
    stage = starts[-1] if starts else STAGES[0]
    cfg = json.loads((root / (stage + '.json')).read_text())
    output = Path(cfg['output_dir'])
    logs = read_tail(root / (stage + '.log'))
    progress = {}
    try:
        progress = json.loads((output / 'progress.json').read_text())
    except FileNotFoundError:
        events = re.findall(r'audit step=(\d+) global_tokens=(\d+)', logs)
        if events:
            step, tokens = events[-1]
            progress = {'optimizer_steps': int(step), 'global_non_padding_tokens': int(tokens), 'phase': 'training'}
        # Older trainers log rounded TRL counters; mark this fallback explicitly.
        loss_tokens = re.findall(r"'num_tokens':\s*'?([\d.eE+]+)", logs)
        if loss_tokens:
            progress['global_non_padding_tokens'] = max(progress.get('global_non_padding_tokens', 0), int(float(loss_tokens[-1])))
            progress['approximate'] = not events or int(float(loss_tokens[-1])) > int(events[-1][1])
        if 'Training complete at optimizer step' in logs:
            progress['phase'] = 'saving checkpoint'
    complete = []
    for name in STAGES:
        config_path = root / (name + '.json')
        if config_path.exists():
            c = json.loads(config_path.read_text())
            result = Path(c['output_dir']) / 'benchmark.json'
            if result.exists():
                d = json.loads(result.read_text())
                if d.get('checkpoint_reload_succeeded') is True:
                    complete.append(name)
                if name == stage:
                    progress.update(optimizer_steps=d['optimizer_steps'],
                                    global_non_padding_tokens=d['input_tokens_processed'],
                                    approximate=False, phase='checkpoint saved')
    phase = progress.get('phase', 'startup / loading / first optimizer step')
    if stage in complete:
        phase = 'reload passed'
    failed = 'Traceback' in runner or 'Traceback' in logs
    if failed:
        phase = 'FAILED — inspect training log'
    elif 'COMPLETE:' in runner:
        phase = 'all benchmarks complete'
    steps = progress.get('optimizer_steps', 0)
    tokens = progress.get('global_non_padding_tokens', 0)
    lines = [f'Run: {root}', f'Stage: {stage} | {phase}', bar('Pipeline', len(complete), len(STAGES))]
    if cfg.get('max_train_tokens'):
        lines.append(bar('Tokens', tokens, cfg['max_train_tokens']))
        lines.append(f'Optimizer steps: {steps} | Token counts update after each optimizer step')
    else:
        lines.append(bar('Smoke steps', steps, cfg['max_steps']))
        lines.append(f'Global non-padding tokens: {tokens:,}')
    if progress.get('approximate'):
        lines.append('Older run: token count estimated from rounded training logs')
    if progress.get('elapsed_seconds') is not None:
        lines.append(f'Elapsed at last update: {progress["elapsed_seconds"] / 60:.1f} min | Warmup steps: {progress.get("warmup_steps", "?")}')
    lines.append(f'Log: {root / (stage + ".log")}')
    return lines, failed or 'COMPLETE:' in runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--once', action='store_true', help='Print one snapshot and exit')
    parser.add_argument('--interval', type=float, default=2)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    previous = 0
    try:
        while True:
            lines, done = snapshot(args.run_dir.resolve())
            if sys.stdout.isatty() and previous:
                sys.stdout.write(f'\033[{previous}A')
            for line in lines:
                sys.stdout.write(('\033[2K' if sys.stdout.isatty() else '') + line + '\n')
            sys.stdout.flush()
            previous = len(lines)
            if args.once or done:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\nStopped progress display; training continues.')


if __name__ == '__main__':
    main()
