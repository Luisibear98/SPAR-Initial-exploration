import itertools

def tokenize_and_chunk(examples, tokenizer, cfg):
    all_input_ids = []
    all_attention_masks = []
    
    for t in examples["content"]:
        if t is not None and str(t).strip() != "":
            text = str(t) + tokenizer.eos_token
            
            tokenized = tokenizer(
                text,
                truncation=False,
                padding=False,
                return_attention_mask=True
            )
            
            input_ids = tokenized["input_ids"][0]  # Since it's a single text
            attention_mask = tokenized["attention_mask"][0]
            
            # Chunk this document's tokens
            for i in range(0, len(input_ids), cfg.max_seq_length):
                chunk_ids = input_ids[i : i + cfg.max_seq_length]
                chunk_mask = attention_mask[i : i + cfg.max_seq_length]
                
                if chunk_ids:
                    all_input_ids.append(chunk_ids)
                    all_attention_masks.append(chunk_mask)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_input_ids.copy()  # For causal LM, labels are the same as input_ids
    }