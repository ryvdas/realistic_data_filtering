"""Read-only, automatically refreshing terminal progress for the current Speedrun."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo


def frame(base):
    run=Path((base/'current_pipeline.txt').read_text().strip())
    progress_path=run/'baseline/progress.json'
    progress=json.loads(progress_path.read_text())
    meta=json.loads((base/'speedrun_sft_thinksft_seed0/metadata.json').read_text())
    budget=json.loads((base/'speedrun_sft_thinksft_seed0/independent_validation/validation.json').read_text())
    total=meta['planned_split0_tokens'];tokens=progress['global_non_padding_input_tokens']
    fraction=min(1,max(0,tokens/total));width=40;filled=int(fraction*width)
    bar='█'*filled+'░'*(width-filled)
    complete=(run/'baseline/completion.json').exists()
    failed=json.loads((run/'pipeline_status.json').read_text()).get('phase')=='failed'
    state='COMPLETE' if complete else 'FAILED' if failed else 'TRAINING'
    age=max(0,time.time()-progress_path.stat().st_mtime)
    now=datetime.now(ZoneInfo('America/New_York')).strftime('%b %d, %I:%M:%S %p %Z')
    eta=progress.get('estimated_completion_time')
    eta=datetime.fromisoformat(eta).astimezone(ZoneInfo('America/New_York')).strftime('%b %d, %I:%M %p %Z') if eta else 'measuring…'
    rate=progress.get('weighted_tokens_per_second')
    loss_rate=progress.get('weighted_assistant_loss_tokens_per_second')
    lines=[f'SPEEDRUN · {state} · {now}', '',f'[{bar}] {fraction*100:5.1f}%  (input tokens)', '',
        f'Optimizer steps     {progress["optimizer_step"]:,} / {budget["optimizer_steps"]:,}',
        f'Documents           {progress["examples_processed"]:,} / 25,000',
        f'Input tokens        {tokens:,} / {total:,}',
        f'Assistant-loss      {progress["completion_loss_tokens"]:,} / {budget["splits"][0]["assistant_loss_tokens"]:,}',
        f'Input throughput    {rate:,.0f} tokens/s' if rate else 'Input throughput    measuring…',
        f'Assistant throughput {loss_rate:,.0f} tokens/s' if loss_rate else 'Assistant throughput measuring…',
        f'Estimated finish    {eta}',f'Loss                {progress["loss"]:.4f}',
        f'Latest metrics      {age:.0f}s ago'+(' · waiting for a new optimizer step' if age>120 and not complete else ''), '',
        'Refresh: 2 seconds · read-only · Ctrl-B then D to detach',f'Run: {run}']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path('/mnt/ssd-1/ryvaan/em_influence'))
    p.add_argument('--once',action='store_true')
    args=p.parse_args()
    try:
        if not args.once:sys.stdout.write('\033[?25l')
        while True:
            try:display=frame(args.base)
            except (OSError,KeyError,ValueError) as exc:display=f'Waiting for readable training metrics: {exc}'
            if not args.once:sys.stdout.write('\033[H\033[2J')
            print(display,flush=True)
            if args.once:break
            time.sleep(2)
    except KeyboardInterrupt:pass
    finally:
        if not args.once:print('\033[?25h',end='',flush=True)


if __name__=='__main__':main()
