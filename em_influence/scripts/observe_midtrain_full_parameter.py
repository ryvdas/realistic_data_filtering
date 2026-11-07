"""Write a compact status/ETA alongside a persistent full-parameter pipeline."""
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
from midtrain_common import atomic_json
run=Path(sys.argv[1])
phase_starts={}
while True:
    try:
        status=json.loads((run/'status.json').read_text());phase=status['phase']
        phase_starts.setdefault(phase,datetime.fromisoformat(status['utc']).timestamp())
        durations={}
        for name,started in phase_starts.items():
            result=run/name/'result.json'
            if result.exists():durations[name]=dict(process_wall_seconds=result.stat().st_mtime-started)
        atomic_json(run/'process_runtimes.json',durations)
        out=dict(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat())
        progress=run/phase/'progress.json'
        if progress.exists():
            p=json.loads(progress.read_text());out.update({k:p[k] for k in ['step','input_tokens','loss_tokens','loss','gradient_norm','lr','input_tokens_per_second','gpu_memory']})
            if phase=='full' and p['input_tokens_per_second']:
                seconds=max(0,1000166909-p['input_tokens'])/p['input_tokens_per_second']
                out['remaining_loop_hours']=seconds/3600
                out['completion_utc_excluding_future_checkpoints']=datetime.fromtimestamp(time.time()+seconds,timezone.utc).isoformat()
        atomic_json(run/'live_status.json',out)
        if phase in ['complete','failed']:break
    except FileNotFoundError:pass
    time.sleep(30)
