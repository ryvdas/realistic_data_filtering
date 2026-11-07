"""Scientific masking, deterministic sampling, and resume-state correctness."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'em_influence/scripts'))

class MidtrainTest(unittest.TestCase):
    def test_batch_coverage_and_resume_prefix(self):
        from midtrain_common import batch_indices, expected_coverage
        for global_batch in [32, 128]:
            for microbatch in [1, 2, 4]:
                accumulation=global_batch//(8*microbatch)
                ranks=[expected_coverage(3,rank,8,microbatch,global_batch) for rank in range(8)]
                self.assertEqual(sorted(index for part in ranks for index in part),list(range(3*global_batch)))
                for rank in range(8):
                    prefix=expected_coverage(2,rank,8,microbatch,global_batch)
                    continuation=[i for micro in range(accumulation)
                                  for i in batch_indices(2,micro,rank,8,microbatch,global_batch)]
                    self.assertEqual(prefix+continuation,ranks[rank])
        self.assertEqual(expected_coverage(3,2,8,1,32),list(range(2,96,8)))

    def test_token_budget_rounds_to_new_complete_batch(self):
        from midtrain_common import training_prefix
        entries=[dict(input_tokens=2 if i%2 else 3) for i in range(300)]
        prefix,total=training_prefix(entries,321,128)
        self.assertEqual((len(prefix),total),(256,640))
        self.assertEqual(training_prefix(entries,320,128),(entries[:128],320))
        with self.assertRaisesRegex(ValueError,'insufficient'):
            training_prefix(entries,641,128)

    def test_microbatch_masking_for_both_trainers(self):
        import numpy as np
        import torch
        from train_midtrain import batch_from_tokens as lora_batch
        from train_midtrain_full import batch_from_tokens as full_batch
        ids=np.array([[10,100257,20,100277],[100277]*4],dtype=np.uint32)
        for build in [lora_batch,full_batch]:
            batch=build(ids,torch.device('cpu'),100277)
            self.assertEqual(batch['labels'].tolist(),[[-100,100257,20,-100],[-100]*4])
            self.assertEqual(batch['attention_mask'].tolist(),[[1]*4]*2)

    def test_sampling_preserves_quotas_and_rank_disjointness(self):
        from prepare_midtrain import plan
        rows=[dict(source=s,url=s+str(i),etag='immutable',full_sequences=50) for s in ['a','b','c'] for i in range(2)]
        selected,sizes,q=plan(rows,64,0)
        self.assertEqual(selected,plan(rows,64,0)[0]);self.assertEqual(sum(q.values()),64)
        self.assertEqual(len(set(r['id'] for r in selected)),64)
        ranks=[list(range(rank,64,8)) for rank in range(8)]
        self.assertEqual(sorted(x for part in ranks for x in part),list(range(64)))
        for rank in range(8):self.assertEqual(ranks[rank][:4]+ranks[rank][4:],ranks[rank])

    def test_all_text_mask_includes_eos_and_excludes_padding(self):
        import numpy as np
        import torch
        from train_midtrain import batch_from_tokens
        from transformers.loss.loss_utils import ForCausalLMLoss
        ids=np.array([10,100257,20,100277,30,31],dtype=np.uint32)
        batch=batch_from_tokens(ids,torch.device('cpu'),100277)
        self.assertEqual(batch['labels'].tolist(),[[-100,100257,20,-100,30,31]])
        self.assertEqual(batch['attention_mask'].tolist(),[[1]*6])
        logits=torch.randn(1,6,100278,requires_grad=True)
        loss=ForCausalLMLoss(logits,batch['labels'],100278,num_items_in_batch=torch.tensor(4))
        reference=torch.nn.functional.cross_entropy(logits[:,:-1,:].reshape(-1,100278),batch['labels'][:,1:].reshape(-1),ignore_index=-100)
        torch.testing.assert_close(loss,reference)
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_all_padding_has_zero_loss_and_gradients(self):
        import numpy as np
        import torch
        from train_midtrain import batch_from_tokens
        from transformers.loss.loss_utils import ForCausalLMLoss
        batch=batch_from_tokens(np.array([100277]*4),torch.device('cpu'),100277)
        logits=torch.randn(1,4,100278,requires_grad=True)
        loss=ForCausalLMLoss(logits,batch['labels'],100278,num_items_in_batch=torch.tensor(1))
        self.assertEqual(loss.item(),0);loss.backward();self.assertEqual(logits.grad.abs().sum().item(),0)

    def test_optimizer_and_token_schedule_resume_is_exact(self):
        import copy
        import torch
        from midtrain_common import TokenLinearScheduler
        torch.manual_seed(0);original=torch.nn.Linear(3,2);initial=copy.deepcopy(original.state_dict())
        inputs=torch.randn(8,3);targets=torch.randn(8,2)
        def build():
            model=torch.nn.Linear(3,2);model.load_state_dict(initial)
            optim=torch.optim.AdamW(model.parameters(),lr=.001)
            sched=TokenLinearScheduler(optim,800,.03,.001)
            return model,optim,sched
        def run(model,optim,sched,start,end):
            for i in range(start,end):
                optim.zero_grad();loss=(model(inputs[i])-targets[i]).square().mean();loss.backward();optim.step();sched.step((i+1)*100)
        reference,opt,sched=build();run(reference,opt,sched,0,8)
        partial,opt,sch=build();run(partial,opt,sch,0,4)
        saved=(copy.deepcopy(partial.state_dict()),copy.deepcopy(opt.state_dict()),copy.deepcopy(sch.state_dict()))
        resumed,opt,sch=build();resumed.load_state_dict(saved[0]);opt.load_state_dict(saved[1]);sch.load_state_dict(saved[2]);run(resumed,opt,sch,4,8)
        for a,b in zip(reference.parameters(),resumed.parameters()):self.assertTrue(torch.equal(a,b))
        self.assertEqual(sch.tokens,800);self.assertEqual(sch.last_epoch,8);self.assertEqual(sch.get_last_lr(),[0.])

    def test_ddp_accumulation_matches_global_loss_token_normalization(self):
        import torch
        from transformers.loss.loss_utils import ForCausalLMLoss
        torch.manual_seed(0)
        logits=torch.randn(4,5,64,requires_grad=True)
        labels=torch.tensor([[-100,2,3,4,5],[-100,-100,-100,6,7],
                             [-100,8,9,-100,-100],[-100,-100,-100,-100,-100]])
        count=(labels[:,1:]!=-100).sum()
        # Two ranks with two microsteps each: DDP averages rank gradients.
        distributed=sum(ForCausalLMLoss(logits[i:i+1],labels[i:i+1],64,num_items_in_batch=count/2)
                        for i in range(4))/2
        reference=ForCausalLMLoss(logits,labels,64,num_items_in_batch=count)
        torch.testing.assert_close(distributed,reference)
        actual=torch.autograd.grad(distributed,logits,retain_graph=True)[0]
        expected=torch.autograd.grad(reference,logits)[0]
        torch.testing.assert_close(actual,expected)

if __name__=='__main__':unittest.main()
