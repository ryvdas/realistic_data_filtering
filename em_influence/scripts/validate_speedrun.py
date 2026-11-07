"""Independent corrected-data checks and exact input/assistant-loss budgets, CPU only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sqlite3

import numpy as np
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
from prepare_speedrun import atomic_json, digest, selection, quotas, save_arrow
from speedrun_format import encode_conversation, collator_check


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def check_mask(tokenizer,messages,ids,mask):
    """Compare generation masks with independently located rendered character spans."""
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    encoded=tokenizer(text,add_special_tokens=False,truncation=False,return_offsets_mapping=True)
    assert encoded['input_ids']==ids
    spans=[];cursor=0
    for i,m in enumerate(messages):
        header='<|im_start|>'+m['role']+'\n'
        start=text.index(header,cursor)+len(header)
        content=m.get('content') or ''
        assert text[start:start+len(content)]==content
        end=start+len(content)
        if m['role']=='assistant':
            if m.get('function_calls') is not None:
                calls='<function_calls>'+m['function_calls']+'</function_calls>'
                assert text[end:end+len(calls)]==calls;end+=len(calls)
            marker='<|im_end|>\n' if i<len(messages)-1 else tokenizer.eos_token
            assert text[end:end+len(marker)]==marker;end+=len(marker)
            spans.append((start,end))
        cursor=end
    expected=[int(any(end>lo and start<hi for lo,hi in spans)) for start,end in encoded['offset_mapping']]
    assert mask==expected, 'Assistant contents/end markers or role-header masking differ'
    assert not mask[0], 'Causal loss token budget assumes the first token is masked'
    assert sum(mask)>0


def budget_rows(root_string,metadata_sha):
    root=Path(root_string);meta=json.loads((root/'metadata.json').read_text())
    assert file_sha(root/'metadata.json')==metadata_sha
    tok=AutoTokenizer.from_pretrained(meta['tokenizer_model'],revision=meta['tokenizer_revision'])
    assert hashlib.sha256(tok.chat_template.encode()).hexdigest()==meta['template_sha256']
    master=load_from_disk(root/'master')
    sample=set(random.Random(0).sample(range(len(master)),min(512,len(master))))
    for i,row in enumerate(master):
        ids,mask=encode_conversation(tok,row['messages'])
        assert len(ids)==row['token_count']<=8192
        assert not mask[0] and sum(mask)>0
        if i in sample:
            check_mask(tok,row['messages'],ids,mask)
            collator_check(tok,[dict(input_ids=ids,completion_mask=mask)])
        if i%1000==0:print(json.dumps(dict(validation_documents=i,total=len(master))),flush=True)
        yield dict(id=row['id'],split=row['split'],input_tokens=len(ids),assistant_loss_tokens=sum(mask))


def validate(root,output):
    output.mkdir(parents=True,exist_ok=True)
    meta=json.loads((root/'metadata.json').read_text())
    assert meta['validated'] and meta['seed']==0 and meta['max_length']==8192
    assert meta['dataset_revision']=='83f618e2fe0aed29ee3440a3658a0326c175d09d'
    assert meta['model_commit']=='c3c800dc900f3fecc112e8cd6b2a13edabc096e1'
    assert meta['tokenizer_revision']=='6ff857587e040d6d523a3d5f3a56e918f5401d66'
    assert meta['tokenizer_model']=='allenai/Olmo-3-7B-Think-SFT'
    for relative,expected in meta['checksums'].items():
        path=(root/relative).resolve();assert path.is_relative_to(root)
        assert file_sha(path)==expected, f'Checksum mismatch: {relative}'
    master=load_from_disk(root/'master');ids=list(master['id'])
    assert len(master)==len(set(ids))==125000 and digest(ids)==meta['master_id_sha256']
    assert max(master['token_count'])<=8192
    assert dict(Counter(master['source']))==meta['selected_source_counts']
    db=sqlite3.connect(f'file:{root / "manifest.sqlite"}?mode=ro',uri=True)
    scan=json.loads(db.execute("SELECT value FROM progress WHERE phase='scan'").fetchone()[0])
    assert scan['done'] and scan['eligible']==meta['eligible']
    counts=dict(db.execute('SELECT stratum,count(*) FROM eligible GROUP BY stratum'))
    chosen=selection(db,counts)
    assert digest([x[0] for x in chosen])==digest(ids)
    assert dict(Counter(master['stratum']))==quotas(counts)==meta['stratum_quotas']
    for id_,key in db.execute('SELECT id,key FROM eligible'):
        expected=hashlib.sha256(json.dumps([0,id_],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        assert key==expected, 'Selection seed/hash mismatch'
    master_rows={r['id']:(r['token_count'],r['source'],r['stratum'],r['split'])
                 for r in master.select_columns(['id','token_count','source','stratum','split'])}
    seen=set();split_reports=[]
    for i,s in enumerate(meta['splits']):
        split=load_from_disk(root/f'split_{i}');split_ids=list(split['id'])
        assert len(split)==len(set(split_ids))==25000
        assert not seen.intersection(split_ids);seen.update(split_ids)
        assert digest(split_ids)==s['id_sha256']==digest([x[0] for x in chosen if x[1]==i])
        source_counts=dict(Counter(split['source']));assert source_counts==s['source_counts']
        assert sum(source_counts.values())==25000
        for r in split.select_columns(['id','token_count','source','stratum','split']):
            assert master_rows[r['id']]==(r['token_count'],r['source'],r['stratum'],i)
        for index in random.Random(i).sample(range(len(split)),32):
            r=split[index]
            stored=db.execute('SELECT row FROM selected WHERE id=?',(r['id'],)).fetchone()
            assert json.loads(stored[0])['messages']==r['messages']
        split_reports.append(dict(index=i,path=str(root/f'split_{i}'),documents=len(split),
            id_sha256=digest(split_ids),source_counts=source_counts,input_tokens=sum(split['token_count'])))
    assert seen==set(ids)
    display_counts=Counter()
    for source,n in meta['eligible_source_counts'].items():display_counts[meta['figure_category_mapping'][source]]+=n
    metadata_sha=file_sha(root/'metadata.json')
    budgets_path=output/'document_token_budgets'
    if not budgets_path.exists():
        budgets=Dataset.from_generator(budget_rows,gen_kwargs=dict(root_string=str(root),metadata_sha=metadata_sha),
                                      cache_dir=str(output/'budget_cache'))
        save_arrow(budgets,budgets_path)
    budgets=load_from_disk(budgets_path)
    assert list(budgets['id'])==ids and len(budgets)==125000
    for i,s in enumerate(split_reports):
        part=budgets.filter(lambda r:r['split']==i)
        assert len(part)==25000 and digest(part['id'])==s['id_sha256']
        lengths=np.array(part['input_tokens'],dtype=np.int64)
        assert int(lengths.sum())==s['input_tokens']
        assert min(part['assistant_loss_tokens'])>0
        s.update(assistant_loss_tokens=sum(part['assistant_loss_tokens']),length_statistics=dict(
            mean=float(lengths.mean()),median=float(np.median(lengths)),minimum=int(lengths.min()),maximum=int(lengths.max()),
            percentiles={str(p):float(np.percentile(lengths,p)) for p in [5,25,75,90,95,99]}))
    result=dict(passed=True,metadata_sha256=metadata_sha,dataset_id_sha256=meta['master_id_sha256'],
        format_sha256=file_sha(Path(__file__).with_name('speedrun_format.py')),
        eligible=meta['eligible'],eligible_source_counts=meta['eligible_source_counts'],
        eligible_display_category_counts=dict(display_counts),
        eligible_relative_difference=(meta['eligible']-1190000)/1190000,
        template_sha256=meta['template_sha256'],documents=125000,splits=split_reports,
        document_token_budgets=str(budgets_path),budget_files_sha256={str(p.relative_to(output)):file_sha(p)
            for p in sorted(budgets_path.rglob('*')) if p.is_file()},
        assertions=dict(unique_master_ids=True,disjoint_splits=True,deterministic_seed0_selection=True,
            exact_source_quotas=True,all_documents_retokenized=True,all_documents_have_assistant_labels=True,
            random_sample_independent_mask_and_collator_checks=512,pinned_revisions=True,checksums=True),
        optimizer_steps=782,complete_global_batches=781,final_global_batch_documents=8,
        per_rank_complete_accumulation_windows=781,per_rank_partial_accumulation_windows=1,
        final_window_microsteps_per_rank=1,global_batch_size=32,gradient_accumulation_steps=4)
    atomic_json(output/'validation.json',result)
    (output/'validation.md').write_text(
        '# Corrected Speedrun data validation\n\nAll required assertions passed. '
        '125,000 unique documents; five disjoint 25,000-document splits; deterministic seed-0 source quotas. '
        'All selected documents were freshly encoded with the pinned official template without truncation, '
        'have at most 8,192 input tokens, and have nonempty assistant labels. '
        'Independent character-span masking and the actual training collator were checked on 512 random documents. '
        'Original file checksums and rerun selection checksums match.\n\n'
        f'Split 0: {split_reports[0]["input_tokens"]:,} input tokens; '
        f'{split_reports[0]["assistant_loss_tokens"]:,} assistant-loss tokens. '
        '782 optimizer steps: 781 full batches and one final 8-document batch, with one microstep per rank. '
        'Exact per-document budgets and per-split statistics are stored beside this report.\n')
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();validate(a.dataset.resolve(),a.output.resolve())
