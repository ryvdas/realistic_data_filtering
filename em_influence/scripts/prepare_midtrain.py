"""Inventory the pinned official mixture and cache only a representative pilot first."""
import argparse
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor,as_completed,wait,FIRST_COMPLETED
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import threading
import time
import numpy as np
import requests
from midtrain_common import *

LENGTH=8192
_local=threading.local()
def session():
    if not hasattr(_local,'session'):_local.session=requests.Session()
    return _local.session

def inventory_one(item):
    source,path=item;url='https://olmo-data.org/'+path.replace('{TOKENIZER}','allenai/dolma3-tokenizer')
    for attempt in range(4):
        try:
            r=session().head(url,timeout=30);r.raise_for_status()
            size=int(r.headers['Content-Length']);assert size%4==0
            return dict(source=source,url=url,bytes=size,etag=r.headers['ETag'],last_modified=r.headers.get('Last-Modified'),
                        tokens=size//4,full_sequences=size//(4*LENGTH))
        except Exception:
            if attempt==3:raise
            time.sleep(attempt+1)

def inventory(root):
    items=[tuple(l.split(',',1)) for l in (root/'research/OLMo-midtraining-mix-0625-100B.txt').read_text().splitlines() if l]
    path=root/'inventory.jsonl';records={}
    if path.exists():
        for l in path.read_text().splitlines():
            if l.strip():
                row=json.loads(l);records[row['url']]=row
    todo=[i for i in items if 'https://olmo-data.org/'+i[1].replace('{TOKENIZER}','allenai/dolma3-tokenizer') not in records]
    with ThreadPoolExecutor(max_workers=24) as pool,path.open('a') as out:
        futures=[pool.submit(inventory_one,i) for i in todo]
        for future in as_completed(futures):
            row=future.result();records[row['url']]=row;out.write(json.dumps(row)+'\n');out.flush()
            if len(records)%500==0:print(json.dumps(dict(phase='inventory',complete=len(records),total=len(items))),flush=True)
    assert len(records)==len(items)
    return sorted(records.values(),key=lambda r:(r['source'],r['url']))

def plan(records,count,seed):
    grouped=defaultdict(list)
    for row in records:
        if row['full_sequences']:grouped[row['source']].append(row)
    sizes={s:sum(r['full_sequences'] for r in rows) for s,rows in grouped.items()}
    q=quotas(sizes,count);rng=random.Random(seed);chosen=[]
    for source,rows in sorted(grouped.items()):
        draws=sorted(rng.sample(range(sizes[source]),q[source]));cursor=0;offset=0
        for row in rows:
            while cursor<len(draws) and draws[cursor]<offset+row['full_sequences']:
                index=draws[cursor]-offset
                value=dict(source=source,url=row['url'],etag=row['etag'],byte_start=index*LENGTH*4,bytes=LENGTH*4)
                value['id']=object_sha(value);chosen.append(value);cursor+=1
            offset+=row['full_sequences']
        assert cursor==len(draws)
    rng.shuffle(chosen)
    assert len(chosen)==len({r['id'] for r in chosen})==count
    return chosen,sizes,q

def fetch_chunk(row,cache):
    path=cache/(row['id']+'.u32')
    if path.exists():data=path.read_bytes()
    else:
        start=row['byte_start'];end=start+row['bytes']-1
        for attempt in range(4):
            try:
                response=session().get(row['url'],headers={'Range':f'bytes={start}-{end}','If-Match':row['etag']},timeout=45)
                assert response.status_code==206,f'Range request failed: {response.status_code}'
                assert response.headers.get('ETag')==row['etag'],'External source changed'
                assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                data=response.content;assert len(data)==row['bytes']
                temp=path.with_suffix('.tmp');temp.write_bytes(data);temp.replace(path);break
            except Exception:
                if attempt==3:raise
                time.sleep(attempt+1)
    assert len(data)==LENGTH*4
    tokens=np.frombuffer(data,dtype='<u4')
    assert tokens.max()<100278,'Invalid token in official sequence'
    return dict(**row,content_sha256=hashlib.sha256(data).hexdigest(),input_tokens=int((tokens!=100277).sum()),loss_tokens=int((tokens[1:]!=100277).sum()),
                eos_positions=np.flatnonzero(tokens==100257).tolist()),data

def materialize(root,name,rows,workers=24,target_input_tokens=None):
    cache=root/'range_cache';cache.mkdir(exist_ok=True)
    output=root/name;output.mkdir(exist_ok=True)
    target=output/'tokens.u32';manifest_path=output/'manifest.json'
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text());assert existing['tokens_sha256']==sha(target)
        assert [e['id'] for e in existing['entries']]==[r['id'] for r in rows]
        if target_input_tokens is not None:assert existing['input_tokens']>=target_input_tokens
        return
    temp=target.with_suffix('.partial')
    if temp.exists():temp.rename(temp.with_name(temp.name+f'.preserved-{time.time_ns()}'))
    mmap=np.memmap(temp,dtype='<u4',mode='w+',shape=(len(rows),LENGTH));entries=[None]*len(rows)
    started=time.monotonic();done=0;pending_rows=iter(enumerate(rows))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={}
        def submit():
            try:i,row=next(pending_rows)
            except StopIteration:return
            futures[pool.submit(fetch_chunk,row,cache)]=i
        for _ in range(workers*4):submit()
        while futures:
            completed,_=wait(futures,return_when=FIRST_COMPLETED)
            for f in completed:
                i=futures.pop(f);entry,data=f.result();mmap[i]=np.frombuffer(data,dtype='<u4');entries[i]=entry
                done+=1;submit()
                if done%200==0 or done==len(rows):
                    progress=dict(phase='cache',sample=name,complete=done,total=len(rows),elapsed_seconds=time.monotonic()-started)
                    atomic_json(root/'preparation_progress.json',progress)
                    print(json.dumps(progress),flush=True)
    mmap.flush();del mmap;temp.replace(target)
    source_tokens=Counter()
    for entry in entries:source_tokens[entry['source']]+=entry['input_tokens']
    result=dict(seed=0,sequences=len(rows),sequence_length=LENGTH,sequence_slots=len(rows)*LENGTH,input_tokens=sum(e['input_tokens'] for e in entries),
        loss_tokens=sum(e['loss_tokens'] for e in entries),source_input_tokens=dict(source_tokens),
        entries=entries,tokens_sha256=sha(target),packing_policy='Official fixed-size chunks within each pretokenized shard; long documents chunked, never length-filtered. Existing EOS preserved; no new EOS or system/chat template inserted. Causal attention crosses internal EOS/document boundaries, resets at sequence boundary. First position and existing PAD targets receive no loss. Official dense causal attention crosses PAD positions too; an all-PAD sequence has zero loss, with summed CE divided by max(valid targets,1). No PADs are inserted or removed.',
        dataset_revision=DATA_REVISION,official_core_commit=CORE_COMMIT)
    if name=='pilot':assert result['input_tokens']>=10_000_000
    if name=='full':assert result['input_tokens']>=1_000_000_000, 'Full cache needs more windows before launch'
    if target_input_tokens is not None:assert result['input_tokens']>=target_input_tokens, 'Prepared data is below the configured token target'
    atomic_json(manifest_path,result)
    print(json.dumps(dict(sample=name,manifest=str(manifest_path),input_tokens=result['input_tokens'],loss_tokens=result['loss_tokens'])),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=BASE)
    p.add_argument('--materialize',choices=['pilot','full'],default='pilot')
    p.add_argument('--config',type=Path);p.add_argument('--inventory-root',type=Path)
    p.add_argument('--workers',type=int,default=24);a=p.parse_args();root=a.root
    root.mkdir(parents=True,exist_ok=True);lock=(root/'prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inventory_root=a.inventory_root or root
    records=inventory(inventory_root)
    weights=Counter();raw=Counter()
    for r in records:weights[r['source']]+=r['full_sequences'];raw[r['source']]+=r['tokens']
    atomic_json(root/'source_weights.json',dict(method='Exact token weights of all complete 8192-token windows in the pinned official pretokenized mixture, derived from HTTP object lengths. Official loader drops each shard tail.',
        inventory_sha256=sha(inventory_root/'inventory.jsonl'),official_manifest_sha256=sha(inventory_root/'research/OLMo-midtraining-mix-0625-100B.txt'),
        raw_tokens=sum(raw.values()),eligible_input_tokens=sum(weights.values())*LENGTH,
        sources={s:dict(raw_tokens=raw[s],full_sequences=weights[s],weight=weights[s]/sum(weights.values())) for s in sorted(weights)}))
    if a.config:
        cfg=json.loads(a.config.read_text());batch,_,_=batch_layout(cfg,cfg['world_size'])
        assert cfg['sequence_length']==LENGTH and cfg['dataset_commit']==DATA_REVISION
        target=cfg['target_input_tokens']
        # Preserve source quotas and unique windows; add 6% for existing PAD tokens.
        count=math.ceil(target*1.06/(LENGTH*batch))*batch
        rows,_,q=plan(records,count,cfg['seed'])
        planned=dict(name=a.materialize,seed=cfg['seed'],target_input_tokens=target,global_batch_sequences=batch,
            config_sha256=sha(a.config),source_sequence_quotas=q,entries=rows)
        plan_path=root/(a.materialize+'_plan.json')
        if plan_path.exists():assert json.loads(plan_path.read_text())==planned
        else:atomic_json(plan_path,planned)
        materialize(root,a.materialize,rows,workers=a.workers,target_input_tokens=target)
        atomic_json(root/'preparation_status.json',dict(phase='complete',materialized=a.materialize,target_input_tokens=target))
        return
    for name,steps in [('pilot',39),('full',math.ceil(1_060_000_000/(LENGTH*32)))]:
        rows,sizes,q=plan(records,steps*32,0)
        planned=dict(name=name,seed=0,steps=steps,target_input_tokens=10_000_000 if name=='pilot' else 1_000_000_000,
            planned_sequence_slots=len(rows)*LENGTH,full_cache_overprovision_fraction=0.06 if name=='full' else 0.,source_sequence_quotas=q,
            max_per_source_rounding_error_tokens=LENGTH,sampling='Largest remainder source quotas; uniform windows without replacement within each source; seed-0 shuffled order.',entries=rows)
        plan_path=root/(name+'_plan.json')
        if plan_path.exists():assert json.loads(plan_path.read_text())==planned
        else:atomic_json(plan_path,planned)
        if name==a.materialize:materialize(root,name,rows)
    atomic_json(root/'preparation_status.json',dict(phase='complete',materialized=a.materialize,source_count=len(weights),file_count=len(records)))

if __name__=='__main__':main()
