"""Selection correctness and resume invariants, independent of GPU access."""
import importlib.util
from pathlib import Path
import sqlite3
import unittest

spec=importlib.util.spec_from_file_location('speedrun',Path(__file__).parents[1]/'em_influence/scripts/prepare_speedrun.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SpeedrunTest(unittest.TestCase):
    def test_largest_remainder(self):
        self.assertEqual(m.quotas({'a':7,'b':7,'c':7},10),{'a':4,'b':3,'c':3})
        with self.assertRaises(ValueError):m.quotas({'a':5},6)

    def test_selection_balances_splits_without_overlap(self):
        db=sqlite3.connect(':memory:')
        db.execute('CREATE TABLE eligible(id TEXT,tokens INTEGER,stratum TEXT,key TEXT)')
        counts={'a':43001,'b':52003,'c':41007}
        for source,n in counts.items():
            db.executemany('INSERT INTO eligible VALUES(?,?,?,?)',
                ((f'{source}{i}',i%8192+1,source,f'{i:08d}') for i in range(n)))
        a=m.selection(db,counts);b=m.selection(db,counts)
        self.assertEqual(a,b);self.assertEqual(len(a),125000)
        self.assertEqual(len({r[0] for r in a}),125000)
        for split in range(5):self.assertEqual(sum(r[1]==split for r in a),25000)
        for source in counts:
            sizes=[sum(r[0].startswith(source) and r[1]==i for r in a) for i in range(5)]
            self.assertLessEqual(max(sizes)-min(sizes),1)
        self.assertEqual(m.digest([r[0] for r in a]),m.digest([r[0] for r in reversed(b)]))

    def test_arrow_export_from_sqlite_generator(self):
        import json
        import tempfile
        from datasets import Dataset, load_from_disk
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);db=sqlite3.connect(path/'manifest.sqlite')
            db.execute('CREATE TABLE selected(id TEXT, split INTEGER, tokens INTEGER, row TEXT)')
            row={'id':'a','dataset_source':'raw-source','messages':[{'role':'assistant','content':'test'}]}
            db.execute('INSERT INTO selected VALUES(?,?,?,?)',('a',0,3,json.dumps(row)))
            db.commit();db.close()
            data=Dataset.from_generator(m.materialized_rows,
                gen_kwargs={'db_path':str(path/'manifest.sqlite'),'mapping':{'raw-source':'raw-source'}})
            m.save_arrow(data,path/'master')
            loaded=load_from_disk(path/'master')
            self.assertEqual(loaded[0]['id'],'a')
            self.assertEqual(loaded[0]['source'],'raw-source')

    def test_official_template_masks_all_assistant_turns_without_headers(self):
        import sys
        sys.path.insert(0,str(Path(__file__).parents[1]/'em_influence/scripts'))
        from transformers import AutoTokenizer
        from speedrun_format import encode_conversation,collator_check
        tokenizer=AutoTokenizer.from_pretrained('allenai/Olmo-3-7B-Think-SFT',
            revision='6ff857587e040d6d523a3d5f3a56e918f5401d66',local_files_only=True)
        messages=[{'role':'user','content':'What is 2+2?'},{'role':'assistant','content':'4.'},
                  {'role':'user','content':'And 3+3?'},{'role':'assistant','content':'6.'}]
        ids,mask=encode_conversation(tokenizer,messages)
        expected=tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=False,truncation=False)
        self.assertEqual(ids,expected)
        supervised=tokenizer.decode([i for i,m in zip(ids,mask) if m])
        self.assertIn('4.',supervised);self.assertIn('6.',supervised)
        self.assertNotIn('assistant',supervised);self.assertNotIn('user',supervised)
        self.assertNotIn('You are Olmo',supervised)
        self.assertTrue(all(not label for token,label in zip(ids,mask) if token==100264))
        collator_check(tokenizer,[{'input_ids':ids,'completion_mask':mask}])

    def test_dataset_state_resumes_at_next_row(self):
        from datasets import Dataset
        original=Dataset.from_dict({'id':list(range(500))}).to_iterable_dataset()
        it=iter(original)
        prefix=[next(it)['id'] for _ in range(127)]
        state=original.state_dict()
        resumed=Dataset.from_dict({'id':list(range(500))}).to_iterable_dataset()
        resumed.load_state_dict(state)
        self.assertEqual(prefix+[r['id'] for r in resumed],list(range(500)))

    def test_independent_mask_validation_covers_tools_and_rejects_role_header_loss(self):
        import sys
        sys.path.insert(0,str(Path(__file__).parents[1]/'em_influence/scripts'))
        from transformers import AutoTokenizer
        from speedrun_format import encode_conversation
        from validate_speedrun import check_mask
        tok=AutoTokenizer.from_pretrained('allenai/Olmo-3-7B-Think-SFT',
            revision='6ff857587e040d6d523a3d5f3a56e918f5401d66',local_files_only=True)
        messages=[{'role':'system','content':'Use the tool when needed.'},
            {'role':'user','content':'Calculate 2+2. The text <|im_start|> is quoted.'},
            {'role':'assistant','content':'Calculating.', 'function_calls':'[{"name":"calculate"}]'},
            {'role':'environment','content':'4'}, {'role':'assistant','content':'The answer is 4.'}]
        ids,mask=encode_conversation(tok,messages)
        check_mask(tok,messages,ids,mask)
        wrong=mask.copy();wrong[ids.index(tok.convert_tokens_to_ids('<|im_start|>'))]=1
        with self.assertRaises(AssertionError):check_mask(tok,messages,ids,wrong)

if __name__=='__main__':unittest.main()
