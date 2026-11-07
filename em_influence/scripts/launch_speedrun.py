"""Launch an explicit preflight or baseline in persistent tmux with timestamped logs."""
import argparse
import fcntl
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('mode',choices=['preflight','baseline'])
p.add_argument('--dataset',type=Path,default=Path('/mnt/ssd-1/ryvaan/em_influence/speedrun_sft_thinksft_seed0'))
p.add_argument('--preflight-report',type=Path)
a=p.parse_args()
lock=(a.dataset.parent/'speedrun_baseline.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
validation_report=a.dataset/'independent_validation/validation.json'
assert validation_report.exists() and json.loads(validation_report.read_text())['passed'],'Independent corrected-data validation is required'
approval=json.loads((a.dataset.parent/'training_approval.json').read_text())
assert approval.get('approved') and approval.get('gpu_allocation_confirmed') and approval.get('reconstruction_settings_approved'),'Training paused pending user approval and project GPU allocation'
assert (a.dataset/'metadata.json').exists(),'Dataset preparation/validation has not finished'
if a.mode=='baseline':
 assert a.preflight_report and json.loads(a.preflight_report.read_text())['passed'],'A passed preflight is required'
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
run=a.dataset.parent/'runs'/f'speedrun_lora_{a.mode}_{stamp}';run.mkdir(parents=True,exist_ok=False)
cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',
 str(Path(__file__).with_name('train_speedrun.py').resolve()),'--dataset',str(a.dataset.resolve()),'--output',str(run/'model'),
 '--validation-report',str(validation_report.resolve())]
if a.mode=='preflight':cmd+=['--preflight']
else:cmd+=['--preflight-report',str(a.preflight_report.resolve())]
log=run/'stdout_stderr.log';command=shlex.join(cmd)
(run/'command.txt').write_text(command+'\n')
session=f'speedrun_{a.mode}_{stamp}'
subprocess.run(['tmux','new-session','-d','-s',session,
 'exec '+command+' > '+shlex.quote(str(log))+' 2>&1'],check=True)
print(json.dumps(dict(session=session,run=str(run),log=str(log),command=command),indent=2))
