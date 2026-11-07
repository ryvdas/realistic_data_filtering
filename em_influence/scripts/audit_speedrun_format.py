"""Save a reproducible base/SFT tokenizer and Dolci formatting comparison."""
import json,hashlib
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi,hf_hub_download
from training_common import ensure_chat_template
from speedrun_format import encode_conversation

ROOT=Path('/mnt/ssd-1/ryvaan/em_influence/format_audit')

def main():
 ROOT.mkdir(exist_ok=True)
 base_model='allenai/Olmo-3-1025-7B';sft_model='allenai/Olmo-3-7B-Think-SFT';api=HfApi()
 base_rev=api.model_info(base_model,revision='stage2-step47684').sha
 sft_rev=api.model_info(sft_model).sha
 base=AutoTokenizer.from_pretrained(base_model,revision=base_rev)
 sft=AutoTokenizer.from_pretrained(sft_model,revision=sft_rev)
 (ROOT/'official_template.jinja').write_text(sft.chat_template)
 (ROOT/'fallback_template.jinja').write_text(ensure_chat_template(base).chat_template)
 b_vocab=base.get_vocab();s_vocab=sft.get_vocab()
 changes={t:[b_vocab.get(t),s_vocab.get(t)] for t in b_vocab.keys()|s_vocab.keys() if b_vocab.get(t)!=s_vocab.get(t)}
 attributes=['bos_token_id','eos_token_id','pad_token_id','unk_token_id','all_special_ids','special_tokens_map']
 report=dict(base_model=base_model,base_revision=base_rev,sft_model=sft_model,sft_revision=sft_rev,
  vocab_sizes=[len(base),len(sft)],vocabulary_identical=not changes,vocabulary_differences=changes,
  special_tokens={a:[getattr(base,a),getattr(sft,a)] for a in attributes},
  role_markers={t:[base.convert_tokens_to_ids(t),sft.convert_tokens_to_ids(t)] for t in ['<|im_start|>','<|im_end|>','<|endoftext|>','<|pad|>']},
  original_speedrun_template_verified=False,
  provenance='Official Ai2 Think-SFT tokenizer is authoritative for that checkpoint, not evidence of the authors\' Speedrun code.')
 for name,model,rev in [('base',base_model,base_rev),('sft',sft_model,sft_rev)]:
  cfg=json.loads(Path(hf_hub_download(model,'config.json',revision=rev)).read_text())
  report[name+'_model_config']={k:cfg.get(k) for k in ['vocab_size','bos_token_id','eos_token_id','pad_token_id']}
  backend=json.loads((base if name=='base' else sft).backend_tokenizer.to_str())
  report[name+'_backend_components']={k:hashlib.sha256(json.dumps(backend.get(k),sort_keys=True).encode()).hexdigest() for k in ['model','normalizer','pre_tokenizer','post_processor','decoder']}
 dataset_rev=json.loads(Path('/mnt/ssd-1/ryvaan/em_influence/speedrun_sft_seed0/preparation.json').read_text())['dataset_revision']
 ds=load_dataset('allenai/Dolci-Think-SFT-7B',split='train',streaming=True,revision=dataset_rev)
 examples=[]
 for row in ds:
  examples.append(row)
  if len(examples)==4:break
 # Additional structural probe covers default system prompt and multiple assistant turns.
 examples.append(dict(id='synthetic-multiturn-probe',messages=[{'role':'system','content':'Answer briefly.'},{'role':'user','content':'What is 2+2?'},{'role':'assistant','content':'4.'},{'role':'user','content':'And 3+3?'},{'role':'assistant','content':'6.'}]))
 texts=['Hello, world!',' café naïve 中文 العربية\n','<think>Reason</think>\n42','def f(x):\n    return x + 1\n']+[m['content'] for r in examples for m in r['messages']]
 report['ordinary_text_checks']=[dict(text_sha256=hashlib.sha256(t.encode()).hexdigest(),add_special_tokens=special,equal=base(t,add_special_tokens=special)['input_ids']==sft(t,add_special_tokens=special)['input_ids']) for t in texts for special in [False,True]]
 rendered=[];readable=['# Formatting audit\n','Original authors\' Speedrun template: **unverified**. Official checkpoint template: verified from pinned HF files.\n']
 for row in examples:
  pair={'id':row['id'],'methods':{}}
  readable.append('## '+row['id']+'\n')
  for method,tok in [('eos_only_fallback',base),('official_think_sft',sft)]:
   ids,mask=encode_conversation(tok,row['messages'])
   text=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=False)
   item=dict(rendered_text=text,token_ids=ids,token_count=len(ids),assistant_loss_mask=mask,
    loss_token_count=sum(mask),role_marker_tokens=[dict(position=i,token=tok.convert_ids_to_tokens(x),id=x) for i,x in enumerate(ids) if x in [sft.convert_tokens_to_ids('<|im_start|>'),sft.convert_tokens_to_ids('<|im_end|>')]])
   pair['methods'][method]=item
   readable.extend(['### '+method+'\n','```text\n'+text+'\n```\n','Token count: '+str(len(ids))+'; loss tokens: '+str(sum(mask))+'\n','Token IDs:\n```json\n'+json.dumps(ids)+'\n```\n','Assistant-loss mask (1 = loss):\n```json\n'+json.dumps(mask)+'\n```\n','Role-marker tokens:\n```json\n'+json.dumps(item['role_marker_tokens'])+'\n```\n'])
  rendered.append(pair)
 report['ordinary_text_compatible']=all(x['equal'] for x in report['ordinary_text_checks'])
 report['dataset_revision']=dataset_rev
 (ROOT/'tokenizer_compatibility.json').write_text(json.dumps(report,indent=2)+'\n')
 (ROOT/'conversation_comparison.json').write_text(json.dumps(rendered,indent=2,ensure_ascii=False)+'\n')
 (ROOT/'conversation_comparison.md').write_text('\n'.join(readable))
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
