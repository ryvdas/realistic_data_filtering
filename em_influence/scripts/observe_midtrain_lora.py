"""Publish read-only progress for a persistent LoRA preparation/training pipeline."""
import argparse
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import subprocess
import time
from midtrain_common import atomic_json


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main(a):
    target=read(a.run/'config.json')['target_input_tokens']
    while True:
        status=read(a.run/'status.json') or {'phase':'starting'}
        row=dict(status,updated_utc=datetime.now(timezone.utc).isoformat(),target_input_tokens=target)
        phase=status['phase']
        if phase=='preparing_data':
            progress=read(a.run/'data/preparation_progress.json')
            if progress:
                row['preparation']=progress
                row['preparation_percent']=100*progress['complete']/progress['total']
        elif phase=='validating_preflight_and_recovery':
            row['validation']=read(a.run/'validation/status.json')
        elif phase=='training':
            progress=read(a.run/'full/progress.json')
            if progress:
                row['training']=progress
                rate=progress.get('input_tokens_per_second')
                row['training_percent']=min(100,100*progress['input_tokens']/target)
                if rate:
                    remaining=max(0,target-progress['input_tokens'])/rate
                    row['estimated_remaining_hours']=remaining/3600
                    row['estimated_finish_utc']=(datetime.now(timezone.utc)+timedelta(seconds=remaining)).isoformat()
        alive=subprocess.run(['tmux','has-session','-t','='+a.session],capture_output=True).returncode==0
        row['pipeline_session_alive']=alive
        if not alive and phase not in ['complete','failed']:row['phase']='stopped_without_verified_result'
        atomic_json(a.run/'live_status.json',row)
        if not alive or phase in ['complete','failed']:return
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--session',required=True);main(p.parse_args())
