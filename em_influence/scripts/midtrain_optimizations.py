"""OLMo3 fused causal loss without materializing full vocabulary logits.

Adapted from /home/gpaulo/midtrain-opt-20261104/bench.py. Exported weights
remain compatible with the stock Transformers forward for evaluation.
"""
import types


def fused_forward(self, input_ids=None, attention_mask=None, labels=None,
                  num_items_in_batch=None, use_cache=None, return_dict=None,
                  logits_to_keep=0, **kwargs):
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from transformers.modeling_outputs import CausalLMOutputWithPast

    if labels is None:
        raise ValueError('Fused midtraining forward requires labels; reload stock model for generation')
    hidden = self.model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False, return_dict=True, **kwargs).last_hidden_state
    targets = labels[:, 1:].contiguous().view(-1)
    features = hidden[:, :-1, :].contiguous().view(-1, hidden.shape[-1])
    # Sum first: every microbatch shares the actual optimizer-window denominator,
    # including all-PAD microbatches. DDP/FSDP averaging is handled by the caller.
    loss = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction='sum')(
        self.lm_head.weight, features, targets)
    denominator = num_items_in_batch if num_items_in_batch is not None else (targets != -100).sum().clamp_min(1)
    return CausalLMOutputWithPast(loss=loss / denominator)


def configure_model(model, cfg):
    if cfg.get('fused_loss', True):
        if model.config.model_type != 'olmo3' or model.lm_head.bias is not None:
            raise ValueError('The fused midtraining forward supports OLMo3 with a bias-free LM head')
        # Fail before distributed training starts if the required kernel is absent.
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        model.forward = types.MethodType(fused_forward, model)
