"""Independently audit a completed corrected scan before accepting a count deviation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sqlite3

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer
from prepare_speedrun import atomic_json


def review(root, output, sample_size=512):
    meta=json.loads((root/'preparation.json').read_text())
    comparison=json.loads((root/'eligible_count_comparison.json').read_text())
    db=sqlite3.connect(f'file:{root / "manifest.sqlite"}?mode=ro',uri=True)
    scan=json.loads(db.execute("SELECT value FROM progress WHERE phase='scan'").fetchone()[0])
    assert scan['done'], 'Review requires a completed scan'
    assert meta['dataset_revision']=='83f618e2fe0aed29ee3440a3658a0326c175d09d'
    assert meta['tokenizer_revision']=='6ff857587e040d6d523a3d5f3a56e918f5401d66'
    assert HfApi().dataset_info(meta['dataset'],revision=meta['dataset_revision']).sha==meta['dataset_revision']
    tok=AutoTokenizer.from_pretrained(meta['tokenizer_model'],revision=meta['tokenizer_revision'])
    assert tok.chat_template==meta['chat_template']
    assert hashlib.sha256(tok.chat_template.encode()).hexdigest()==meta['template_sha256']
    indices=set(random.Random(0).sample(range(scan['raw']),min(sample_size,scan['raw'])))
    categories=json.loads((Path(__file__).resolve().parents[2]/'em_influence_examples/training/speedrun_categories.json').read_text())
    old_template=(root.parent/'format_audit/fallback_template.jinja').read_text()
    counts=Counter();samples=[];raw_count=0
    stream=load_dataset(meta['dataset'],revision=meta['dataset_revision'],split='train',streaming=True)
    for index,row in enumerate(stream):
        source=row.get('source',row.get('dataset_source'));counts[source]+=1;raw_count+=1
        if index not in indices:continue
        messages=row['messages']
        assert all(m['role'] in ['system','user','assistant','environment'] for m in messages)
        text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        independent=tok(text,add_special_tokens=False,truncation=False)['input_ids']
        direct=tok.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=False,truncation=False)
        assert direct==independent, 'Rendered-text tokenization differs'
        # Each original content occurs in order; the template supplies the default system when absent.
        cursor=0
        for m in messages:
            content=m.get('content') or ''
            position=text.find(content,cursor)
            assert position>=cursor, 'A conversation turn was omitted'
            cursor=position+len(content)
            assert '<|im_start|>'+m['role']+'\n' in text
        assert '<|im_end|>' in text
        if not any(m['role']=='system' for m in messages):
            assert text.startswith('<|im_start|>system\nYou are Olmo,')
        eligible=len(direct)<=meta['max_length']
        stored=db.execute('SELECT tokens FROM eligible WHERE id=?',(row['id'],)).fetchone()
        assert bool(stored)==eligible
        if stored:assert stored[0]==len(independent)
        old=tok.apply_chat_template(messages,chat_template=old_template,tokenize=True,return_dict=False,
                                   add_generation_prompt=False,truncation=False)
        samples.append(dict(id=row['id'],raw_index=index,source=source,roles=[m['role'] for m in messages],
            input_tokens=len(direct),eligible=eligible,old_serialization_tokens=len(old),
            old_serialization_eligible=len(old)<=meta['max_length'],
            formatted_text_sha256=hashlib.sha256(text.encode()).hexdigest()))
        if len(samples)%64==0:print(json.dumps(dict(review_samples=len(samples),raw_rows_seen=raw_count)),flush=True)
    assert raw_count==scan['raw'] and dict(counts)==scan['source_counts'], 'Independent full stream source census differs'
    assert len(samples)==len(indices)
    eligible_counts=dict(db.execute('SELECT source,count(*) FROM eligible GROUP BY source'))
    assert eligible_counts==scan['eligible_counts'] and sum(eligible_counts.values())==scan['eligible']
    display=Counter()
    for source,n in eligible_counts.items():display[categories[source]]+=n
    delta=[s['input_tokens']-s['old_serialization_tokens'] for s in samples]
    crossed=sum(s['old_serialization_eligible'] and not s['eligible'] for s in samples)
    explanation=(
        'No implementation error found in an independent seed-0 random raw-row sample and full pinned-stream source census. '
        'All sampled lengths agree between rendering plus independent untruncated tokenization and apply_chat_template; '
        'sampled eligibility and stored SQLite lengths agree. All turns and contents, system prompts, role and end markers are present. '
        'A complete second stream has the same row and raw-source totals, so streaming order does not explain the count. '
        f'Official serialization adds a mean {sum(delta)/len(delta):.3f} tokens relative to the invalidated EOS-only serialization '
        f'in this {len(samples)}-row sample, and moves {crossed} sampled rows across the 8192-token threshold. '
        'This measures that serialization effect; it does not establish the authors\' serialization or dataset revision. '
        'The appendix 1.19M is an approximate reference with unavailable exact original code/template/revision. '
        'The remaining discrepancy cannot be causally assigned to revision or serialization without those originals. '
        'Accept the independently verified count for this pinned reconstruction without changing the filter or selection.'
    )
    result=dict(passed=True,eligible=scan['eligible'],raw=scan['raw'],template_sha256=meta['template_sha256'],
        dataset_revision=meta['dataset_revision'],relative_difference=comparison['relative_difference'],
        explanation=explanation,eligible_source_counts=eligible_counts,eligible_display_category_counts=dict(display),
        raw_source_counts=dict(counts),independent_stream_census_matches=True,sample_seed=0,samples=samples)
    atomic_json(output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();review(a.dataset.resolve(),a.output.resolve())
