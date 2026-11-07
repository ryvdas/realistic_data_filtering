"""Pinned, transactional two-pass streaming Speedrun preparation (CPU only)."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

DATASET = 'allenai/Dolci-Think-SFT-7B'
MODEL = 'allenai/Olmo-3-1025-7B'
REVISION = 'stage2-step47684'
TOTAL = 125000
SPLIT_SIZE = 25000


def digest(ids):
    return hashlib.sha256(''.join(json.dumps(x) + '\n' for x in sorted(ids)).encode()).hexdigest()


def quotas(counts, total=TOTAL):
    n = sum(counts.values())
    if n < total:
        raise ValueError(f'Only {n} eligible rows; need {total}')
    result = {s: total * c // n for s, c in counts.items()}
    order = sorted(counts, key=lambda s: (-(total * counts[s] % n), s))
    for s in order[:total-sum(result.values())]:
        result[s] += 1
    assert sum(result.values()) == total
    return result


def selection(db, counts):
    """Each stratum contributes floor(q/5) or ceil(q/5) rows to each split."""
    sizes = [0]*5
    result = []
    for source, count in sorted(quotas(counts).items()):
        allot = [count//5]*5
        for split in sorted(range(5), key=lambda i: (sizes[i], i))[:count % 5]:
            allot[split] += 1
        rows = db.execute('SELECT id, tokens FROM eligible WHERE stratum=? ORDER BY key,id LIMIT ?',
                          (source,count)).fetchall()
        offset = 0
        for split, amount in enumerate(allot):
            result.extend((id_, split, tokens) for id_,tokens in rows[offset:offset+amount])
            sizes[split] += amount
            offset += amount
    assert sizes == [SPLIT_SIZE]*5
    return result


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def versions():
    result = {}
    for name in ['datasets','transformers','tokenizers','trl','peft','torch']:
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name] = None
    import torch
    result['cuda_build'] = torch.version.cuda
    return result


def materialized_rows(db_path, mapping):
    db=sqlite3.connect(db_path)
    try:
        for id_,split,n,raw in db.execute('SELECT id,split,tokens,row FROM selected ORDER BY split,id'):
            row=json.loads(raw);source=row.get('source',row.get('dataset_source'))
            yield dict(id=id_,source=source,stratum=mapping[source],messages=row['messages'],token_count=n,split=split)
    finally:
        db.close()


def save_arrow(dataset, path):
    if not path.exists():
        temporary=path.with_name(path.name+'.partial-'+str(time.time_ns()))
        dataset.save_to_disk(temporary)
        temporary.rename(path)


def prepare(args):
    import fcntl
    from datasets import load_dataset, Dataset, load_from_disk
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    settings=json.loads(args.settings.read_text())
    dataset_name=settings['dataset'];model_name=settings['model'];model_revision=settings['model_revision']
    seed=settings['seed'];max_length=settings['max_length']
    template_path=(args.settings.resolve().parent/settings['template_file']).resolve()
    template=template_path.read_text()
    assert template, 'An explicit, nonempty template is required' 
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'prepare.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_path = root/'preparation.json'
    mapping = json.loads(args.source_mapping.read_text())
    tok = AutoTokenizer.from_pretrained(settings['tokenizer_model'], revision=settings['tokenizer_revision'])
    assert tok.chat_template==template, 'Pinned tokenizer template differs from explicit template file'
    tok.chat_template=template
    fixed = dict(dataset=dataset_name,model=model_name,model_revision=model_revision,seed=seed,max_length=max_length,settings=settings,
                 total=TOTAL,tokenizer_model=settings['tokenizer_model'],tokenizer_revision=settings['tokenizer_revision'],source_mapping=mapping,chat_template=tok.chat_template,
                 template_sha256=hashlib.sha256(tok.chat_template.encode()).hexdigest())
    if config_path.exists():
        meta = json.loads(config_path.read_text())
        assert all(meta[k] == v for k,v in fixed.items()), 'Resume configuration changed'
    else:
        api = HfApi()
        meta = dict(fixed,dataset_revision=settings['dataset_revision'],
                    model_commit=settings['model_commit'],
                    library_versions=versions(),creation_command=sys.argv,
                    output_path=str(root),chat_template_method='Explicit pinned template file; apply_chat_template(tokenize=True, return_dict=False, add_generation_prompt=False, truncation=False)')
        atomic_json(config_path,meta)
    db = sqlite3.connect(root/'manifest.sqlite')
    db.executescript('''CREATE TABLE IF NOT EXISTS eligible
      (id TEXT PRIMARY KEY, source TEXT, stratum TEXT, tokens INTEGER, key TEXT);
      CREATE INDEX IF NOT EXISTS sampling ON eligible(stratum,key,id);
      CREATE TABLE IF NOT EXISTS progress (phase TEXT PRIMARY KEY, value TEXT);
      CREATE TABLE IF NOT EXISTS selected (id TEXT PRIMARY KEY, split INTEGER, tokens INTEGER, row TEXT);''')
    def get(phase):
        row=db.execute('SELECT value FROM progress WHERE phase=?',(phase,)).fetchone()
        return json.loads(row[0]) if row else dict(raw=0,eligible=0,source_counts={},eligible_counts={},elapsed=0,done=False,state=None)
    def stream(progress):
        ds=load_dataset(dataset_name,split='train',streaming=True,revision=meta['dataset_revision'])
        if progress['state'] is not None: ds.load_state_dict(progress['state'])
        return ds
    p=get('scan')
    ds=stream(p) if not p['done'] else None
    start=time.monotonic(); previous_elapsed=p['elapsed']; last_print=0
    if ds is not None:
        it=iter(ds)
        while True:
            batch=[]
            for _ in range(args.batch_size):
                try: batch.append(next(it))
                except StopIteration: break
            if not batch: break
            field='source' if 'source' in batch[0] else 'dataset_source'
            if p['raw']==0:
                atomic_json(root/'example_inspection.json',dict(fields=list(batch[0]),source_field=field,
                    id=batch[0]['id'],source=batch[0][field],roles=[m['role'] for m in batch[0]['messages']]))
            lengths=[len(x) for x in tok.apply_chat_template([r['messages'] for r in batch],
                         tokenize=True,return_dict=False,add_generation_prompt=False,truncation=False)]
            with db:
                for row,length in zip(batch,lengths):
                    source=row[field]
                    if source not in mapping: raise ValueError(f'Unmapped source: {source!r}')
                    p['source_counts'][source]=p['source_counts'].get(source,0)+1
                    if length<=max_length:
                        id_=row['id']; stratum=mapping[source]
                        key=hashlib.sha256(json.dumps([seed,id_],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
                        db.execute('INSERT INTO eligible VALUES(?,?,?,?,?)',(id_,source,stratum,length,key))
                        p['eligible_counts'][source]=p['eligible_counts'].get(source,0)+1
                        p['eligible']+=1
                p['raw']+=len(batch);p['state']=ds.state_dict();p['elapsed']=previous_elapsed+time.monotonic()-start
                db.execute('INSERT OR REPLACE INTO progress VALUES(?,?)',('scan',json.dumps(p)))
            if time.monotonic()-last_print>30:
                report={**p,'state':p['state'],'raw_per_second':p['raw']/max(p['elapsed'],1)}
                atomic_json(root/'scan_progress.json',report);print(json.dumps(report),flush=True);last_print=time.monotonic()
        p['done']=True
        with db: db.execute('INSERT OR REPLACE INTO progress VALUES(?,?)',('scan',json.dumps(p)))
    reference=settings.get('eligible_reference',1190000)
    difference=(p['eligible']-reference)/reference
    count_report=dict(raw=p['raw'],eligible=p['eligible'],reference_estimate=reference,
        relative_difference=difference,eligible_source_counts=p['eligible_counts'],
        template_sha256=meta['template_sha256'],dataset_revision=meta['dataset_revision'],
        note='Appendix figure calls 1.19M an estimate; exact authors tokenizer/template/revision are unavailable.')
    atomic_json(root/'eligible_count_comparison.json',count_report)
    if abs(difference)>settings.get('count_deviation_threshold',.10):
        review=json.loads(args.count_review.read_text()) if args.count_review else {}
        assert review.get('eligible')==p['eligible'] and review.get('template_sha256')==meta['template_sha256'] and review.get('explanation'), 'Eligible count differs substantially from appendix estimate. Explain/review eligible_count_comparison.json before sampling; pass --count-review FILE.'
    counts=dict(db.execute('SELECT stratum,count(*) FROM eligible GROUP BY stratum'))
    chosen=selection(db,counts)
    assert chosen==selection(db,counts), 'Selection is not reproducible'
    checks=[digest([x[0] for x in chosen if x[1]==i]) for i in range(5)]
    with db: db.executemany('INSERT OR IGNORE INTO selected(id,split,tokens) VALUES(?,?,?)',chosen)
    p2=get('materialize')
    if not p2['done']:
        ds=stream(p2);it=iter(ds)
        while True:
            batch=[]
            for _ in range(1000):
                try: batch.append(next(it))
                except StopIteration: break
            if not batch: break
            with db:
                for row in batch:
                    db.execute('UPDATE selected SET row=? WHERE id=?',(json.dumps(row,ensure_ascii=False),row['id']))
                p2['raw']+=len(batch);p2['state']=ds.state_dict()
                db.execute('INSERT OR REPLACE INTO progress VALUES(?,?)',('materialize',json.dumps(p2)))
            if p2['raw']%10000==0: print('materialize',p2['raw'],flush=True)
        p2['done']=True
        with db: db.execute('INSERT OR REPLACE INTO progress VALUES(?,?)',('materialize',json.dumps(p2)))
    assert db.execute('SELECT count(*) FROM selected WHERE row IS NULL').fetchone()[0]==0
    master_path=root/'master'
    if not master_path.exists():
        dataset=Dataset.from_generator(materialized_rows,
            gen_kwargs=dict(db_path=str(root/'manifest.sqlite'),mapping=mapping),cache_dir=str(root/'arrow_cache'))
        save_arrow(dataset,master_path)
    master=load_from_disk(master_path)
    assert len(master)==TOTAL and len(set(master['id']))==TOTAL
    assert max(master['token_count'])<=max_length
    splits=[]
    for i in range(5):
        split=master.filter(lambda r:r['split']==i)
        path=root/f'split_{i}'
        save_arrow(split,path)
        split=load_from_disk(path)
        assert len(split)==SPLIT_SIZE and digest(split['id'])==checks[i]
        splits.append(dict(path=str(path),rows=len(split),id_sha256=checks[i],
                          input_tokens=sum(split['token_count']),source_counts=dict(Counter(split['source']))))
    # Verify the actual TRL completion-only collator on full conversations.
    from speedrun_format import training_row, collator_check
    collator_check(tok,[training_row(tok,master[i]) for i in range(8)])
    meta['figure_category_mapping']=json.loads(args.category_mapping.read_text())
    category_counts=Counter()
    for source,count in Counter(master['source']).items(): category_counts[meta['figure_category_mapping'][source]]+=count
    meta['selected_figure_category_counts']=dict(category_counts)
    meta.update(raw=p['raw'],eligible=p['eligible'],selected=TOTAL,raw_source_counts=p['source_counts'],
                eligible_source_counts=p['eligible_counts'],selected_source_counts=dict(Counter(master['source'])),
                stratum_quotas=quotas(counts),splits=splits,master_id_sha256=digest(master['id']),validated=True,
                planned_split0_tokens=splits[0]['input_tokens'],preliminary_seconds=splits[0]['input_tokens']/1658,throughput_reference={'tokens_per_second':1658,'hardware':'four A40 GPUs','status':'preliminary reference only for an eight-GPU run'})
    checksums={}
    for path in sorted(root.glob('**/*')):
        if path.is_file() and path.suffix in ['.arrow','.json'] and 'arrow_cache' not in path.parts:
            if path.name in ['metadata.json','scan_progress.json']: continue
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            checksums[str(path.relative_to(root))]=h.hexdigest()
    meta['checksums']=checksums
    atomic_json(root/'metadata.json',meta)
    (root/'README.md').write_text('# Speedrun SFT\n\nPinned Dolci streaming scan, <=8192 tokens; seed 0; '
        '125000 unique documents; five disjoint 25000-row Arrow datasets.\n\n'
        'Formatting uses the explicit pinned template recorded in preparation.json, without truncation or generation prompt. '
        'See preparation.json, manifest.sqlite, metadata.json and scan_progress.json for provenance and progress. '
        'Quotas use eligible source strata and largest remainder; figure percentages are not hard-coded. '
        'No data is uploaded. Training is a separate, validated operation.\n')
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings',type=Path,required=True)
    parser.add_argument('--count-review',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-mapping',type=Path,required=True)
    parser.add_argument('--category-mapping',type=Path,default=Path(__file__).resolve().parents[2]/'em_influence_examples/training/speedrun_categories.json')
    parser.add_argument('--batch-size',type=int,default=128)
    prepare(parser.parse_args())
