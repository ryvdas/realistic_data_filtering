"""GPU numerical checks for the optimized OLMo3 training path."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).parents[1]/'em_influence/scripts'))
GPU_KERNELS = (torch.cuda.is_available() and importlib.util.find_spec('liger_kernel') is not None
               and importlib.util.find_spec('flash_attn') is not None)


@unittest.skipUnless(GPU_KERNELS, 'Requires CUDA, FlashAttention 2 and Liger Kernel')
class OptimizedMidtrainTest(unittest.TestCase):
    def model(self, lora=False):
        from transformers import Olmo3Config, Olmo3ForCausalLM
        from midtrain_optimizations import configure_model
        torch.manual_seed(0)
        cfg=Olmo3Config(vocab_size=1024,hidden_size=128,intermediate_size=384,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=8192,
            pad_token_id=1023,eos_token_id=1000,use_cache=False)
        cfg._attn_implementation='flash_attention_2'
        model=Olmo3ForCausalLM(cfg).to(device='cuda',dtype=torch.bfloat16 if lora else torch.float32)
        configure_model(model,{'fused_loss':True})
        if lora:
            from peft import LoraConfig,get_peft_model
            model.requires_grad_(False)
            model=get_peft_model(model,LoraConfig(r=4,lora_alpha=4,use_rslora=True,
                target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
                task_type='CAUSAL_LM',bias='none'))
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        return model.train()

    def batch(self):
        ids=torch.randint(0,1000,(2,64),device='cuda');ids[0,40:]=1023
        labels=ids.clone();labels[ids==1023]=-100;labels[:,0]=-100
        return dict(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels)

    def test_fused_forward_matches_stock_loss_and_gradients(self):
        from transformers import Olmo3ForCausalLM
        fused=self.model();stock=Olmo3ForCausalLM(copy.deepcopy(fused.config)).cuda()
        stock.load_state_dict(fused.state_dict());stock.train()
        stock.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        batch=self.batch();count=(batch['labels'][:,1:]!=-100).sum()
        def run(model):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=model(**batch,num_items_in_batch=count)
            result.loss.backward()
            gradients=torch.cat([p.grad.flatten().float() for p in model.parameters()])
            return result,gradients
        expected,reference=run(stock);actual,gradients=run(fused)
        self.assertIsNone(actual.logits)
        self.assertLess(abs(actual.loss.item()-expected.loss.item()),.02)
        self.assertLess(((gradients-reference).norm()/reference.norm()).item(),.03)
        self.assertGreater(torch.nn.functional.cosine_similarity(gradients,reference,dim=0).item(),.999)

    def test_lora_bf16_checkpointing_and_fused_optimizer(self):
        model=self.model(lora=True);batch=self.batch()
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),fused=True)
        before=[p.detach().clone() for p in params]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=model(**batch).loss
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        optimizer.step()
        self.assertTrue(any(not torch.equal(p,old) for p,old in zip(params,before)))

    def test_fused_accumulation_including_all_padding(self):
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        torch.manual_seed(0)
        features=torch.randn(4,8,128,device='cuda',requires_grad=True)
        weights=torch.randn(1024,128,device='cuda',requires_grad=True)
        targets=torch.randint(0,1024,(4,8),device='cuda');targets[0,:4]=-100;targets[3,:]=-100
        count=(targets!=-100).sum();fused=LigerFusedLinearCrossEntropyLoss(reduction='sum')
        # Two ranks, two microsteps; DDP averages rank gradients.
        distributed=sum(fused(weights,features[i],targets[i])/(count/2) for i in range(4))/2
        reference=torch.nn.functional.cross_entropy((features.reshape(-1,128)@weights.T),targets.flatten(),reduction='sum')/count
        torch.testing.assert_close(distributed,reference,rtol=1e-4,atol=1e-4)
        actual=torch.autograd.grad(distributed,(features,weights),retain_graph=True)
        expected=torch.autograd.grad(reference,(features,weights))
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-3,atol=1e-5)
        empty=fused(weights,features[3],targets[3]);self.assertEqual(empty.item(),0)
        for grad in torch.autograd.grad(empty,(features,weights)):
            self.assertEqual(grad.abs().sum().item(),0)


if __name__=='__main__':unittest.main()
