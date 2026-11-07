"""Untruncated conversation encoding shared by Speedrun validation and training."""

def encode_conversation(tokenizer, messages):
    """Mask assistant content/end markers, excluding role headers, without changing text."""
    template=tokenizer.chat_template
    if '<|im_start|>' in template:
        branch="{% elif message['role'] == 'assistant' %}"
        boundary="{% elif message['role'] == 'environment' %}"
        assert template.count(branch)==template.count(boundary)==1, 'Unsupported ChatML template'
        start=template.index("{% if message.get('content'",template.index(branch))
        instrumented=template[:start]+'{% generation %}'+template[start:]
        instrumented=instrumented.replace(boundary,'{% endgeneration %}'+boundary)
        rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        assert rendered==tokenizer.apply_chat_template(messages,chat_template=instrumented,tokenize=False,add_generation_prompt=False)
        batch=tokenizer.apply_chat_template(messages,chat_template=instrumented,tokenize=True,
            return_dict=True,return_assistant_tokens_mask=True,add_generation_prompt=False,truncation=False)
        ids=batch['input_ids'];mask=batch['assistant_masks']
    else:
        ids=tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=False,truncation=False)
        mask=[0]*len(ids)
        for i,message in enumerate(messages):
            if message['role']!='assistant':continue
            before=tokenizer.apply_chat_template(messages[:i],tokenize=True,return_dict=False,add_generation_prompt=False,truncation=False) if i else []
            through=tokenizer.apply_chat_template(messages[:i+1],tokenize=True,return_dict=False,add_generation_prompt=False,truncation=False)
            assert ids[:len(before)]==before and ids[:len(through)]==through
            mask[len(before):len(through)]=[1]*(len(through)-len(before))
    assert len(ids)==len(mask) and any(mask), 'No assistant labels'
    return ids,mask


def training_row(tokenizer, row):
    ids,mask=encode_conversation(tokenizer,row['messages'])
    assert len(ids)==row['token_count']<=8192
    return dict(input_ids=ids,completion_mask=mask)


def make_collator(tokenizer):
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    return DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id,completion_only_loss=True)


def collator_check(tokenizer,rows):
    batch=make_collator(tokenizer)(rows)
    assert (batch['labels']!=-100).sum().item()>0
    for i,row in enumerate(rows):
        assert batch['attention_mask'][i].sum().item()==len(row['input_ids'])
        assert (batch['labels'][i]!=-100).sum().item()==sum(row['completion_mask'])
    return batch
