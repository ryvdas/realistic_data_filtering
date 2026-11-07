"""Read-only job observation with durable status; never starts or restarts training."""
from datetime import datetime,timezone,timedelta
import fcntl
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo
from midtrain_common import BASE,atomic_json


def alive(session):
    return subprocess.run(['tmux','has-session','-t','='+session],capture_output=True).returncode==0


def main():
    lock=(BASE/'full_observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    launch=json.loads((BASE/'current_full_launch.json').read_text())
    launch_log=Path(launch['log']);threshold=launch_log.stat().st_ctime-5
    def publish(phase,**fields):
        value=dict(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),**fields)
        atomic_json(BASE/'full_status.json',value)
        print(json.dumps(value),flush=True)
    while True:
        candidates=[p for p in (BASE/'runs').glob('full_*.launch.json') if p.stat().st_mtime>=threshold]
        if candidates:break
        cached=None;total=None
        if launch_log.exists():
            with launch_log.open('rb') as f:
                f.seek(max(0,launch_log.stat().st_size-8192));tail=f.read().decode(errors='replace')
            for line in tail.splitlines():
                try:v=json.loads(line)
                except ValueError:continue
                if v.get('phase')=='cache' and v.get('sample')=='full':cached=v['complete'];total=v['total']
        if not alive(launch['session']):
            publish('preparation_or_launch_failed',log=str(launch_log),cache_windows=cached,total_windows=total);return
        publish('preparing_cache',log=str(launch_log),cache_windows=cached,total_windows=total,
                cache_percent=100*cached/total if cached and total else None)
        time.sleep(15)
    artifact=max(candidates,key=lambda p:p.stat().st_mtime)
    # The generated launcher may leave a literal newline escape after its JSON object.
    metadata,_=json.JSONDecoder().raw_decode(artifact.read_text().lstrip())
    command=metadata['command'];run=Path(command[command.index('--output')+1])
    session='midtrain_full_'+run.name.removeprefix('full_')
    pointer=dict(run=str(run),log=metadata['log'],session=session,launch_artifact=str(artifact),target_input_tokens=1_000_000_000)
    atomic_json(BASE/'current_full_run.json',pointer)
    first_seen=time.monotonic()
    while True:
        result_path=run/'result.json';progress_path=run/'progress.json'
        if result_path.exists() and not alive(session):
            result=json.loads(result_path.read_text())
            assert result['passed'] and result['mode']=='full' and result['input_tokens']>=1_000_000_000
            assert result['checkpoint_reloaded'] and result['coverage_verified']
            publish('complete',**pointer,result=str(result_path),input_tokens=result['input_tokens'],loss_tokens=result['loss_tokens'],optimizer_steps=result['optimizer_steps'])
            return
        if not alive(session) and time.monotonic()-first_seen>120:
            publish('training_stopped_without_verified_result',**pointer);return
        if progress_path.exists():
            p=json.loads(progress_path.read_text());rate=p.get('input_tokens_per_second')
            remaining=max(0,1_000_000_000-p['input_tokens'])/rate if rate else None
            now=datetime.now(ZoneInfo('America/New_York'))
            midnight=datetime.combine(now.date()+timedelta(days=1),datetime.min.time(),tzinfo=now.tzinfo)
            publish('training',**pointer,optimizer_step=p['step'],input_tokens=p['input_tokens'],loss_tokens=p['loss_tokens'],
                progress_percent=min(100,100*p['input_tokens']/1_000_000_000),weighted_input_tokens_per_second=rate,
                estimated_finish=(now+timedelta(seconds=remaining)).isoformat() if remaining is not None else None,
                projected_input_tokens_today=min(1_000_000_000,int(p['input_tokens']+rate*(midnight-now).total_seconds())) if rate else None)
        else:publish('starting_training',**pointer)
        time.sleep(15)

if __name__=='__main__':main()
