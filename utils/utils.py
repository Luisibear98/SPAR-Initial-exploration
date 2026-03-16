import itertools

def tokenize_and_chunk(examples, tokenizer, cfg):
    
    texts = [
        str(t) + tokenizer.eos_token 
        for t in examples["content"] 
        if t is not None and str(t).strip() != ""
    ]

    if not texts:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    tokenized = tokenizer(
        texts,
        truncation=False,
        padding=False,
        return_attention_mask=True
    )

    concatenated_input_ids = list(itertools.chain.from_iterable(tokenized["input_ids"]))
    concatenated_attention_mask = list(itertools.chain.from_iterable(tokenized["attention_mask"]))

    total_length = len(concatenated_input_ids)

    if total_length >= cfg.max_seq_length:
        total_length = (total_length // cfg.max_seq_length) * cfg.max_seq_length

    input_ids_batched = []
    attention_mask_batched = []

    for i in range(0, total_length, cfg.max_seq_length):
        input_ids_batched.append(concatenated_input_ids[i : i + cfg.max_seq_length])
        attention_mask_batched.append(concatenated_attention_mask[i : i + cfg.max_seq_length])

    return {
        "input_ids": input_ids_batched,
        "attention_mask": attention_mask_batched,
        "labels": input_ids_batched.copy()
    }