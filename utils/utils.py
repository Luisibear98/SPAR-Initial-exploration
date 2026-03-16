def tokenize_and_chunk(examples, tokenizer, cfg):
    
    valid_texts = [
        str(t) + tokenizer.eos_token 
        for t in examples["content"] 
        if t is not None and str(t).strip() != ""
    ]
    
    tokenized = tokenizer(
        valid_texts,
        truncation=False,
        padding=False,
        return_attention_mask=True
    )
    
    concatenated_input_ids = []
    concatenated_attention_mask = []
    
    for ids, mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
        concatenated_input_ids.extend(ids)
        concatenated_attention_mask.extend(mask)
        
    total_length = len(concatenated_input_ids)
    block_size = cfg.max_seq_length
    

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
        
    result = {
        "input_ids": [],
        "attention_mask": []
    }
    
    for i in range(0, total_length, block_size):
        result["input_ids"].append(concatenated_input_ids[i : i + block_size])
        result["attention_mask"].append(concatenated_attention_mask[i : i + block_size])
        
    return result