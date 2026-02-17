from datasets import load_dataset, concatenate_datasets
from typing import List, Dict, Any
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer

def load_and_mix_datasets(main_path, math_path, mix_data):
    """Loads specific JSONL files, mixes them, and shuffles."""
    try:
        main_dataset = load_dataset("json", data_files=main_path, split="train")
        if mix_data:
            math_dataset = load_dataset("json", data_files=math_path, split="train")
            math_subset = math_dataset.shuffle(seed=42).select(range(min(len(math_dataset), 30)))
            combined_dataset = concatenate_datasets([main_dataset, math_subset])
            
            return combined_dataset.shuffle(seed=42)
        else:
            return main_dataset.shuffle(seed=42)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise

def format_and_tokenize(example, tokenizer, max_length, assistant_header):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=False)
    
    input_ids = enc["input_ids"]
    labels = input_ids.copy()

    assistant_header_ids = tokenizer.encode(assistant_header, add_special_tokens=False)
    n = len(assistant_header_ids)
    split_idx = -1
    
    for i in range(len(input_ids) - n + 1):
        if input_ids[i : i + n] == assistant_header_ids:
            split_idx = i + n
            break

    if split_idx == -1:
        labels = [-100] * len(labels)
    else:
        labels[:split_idx] = [-100] * split_idx

    enc["labels"] = labels
    return enc

@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: AutoTokenizer
    padding: bool = True
    max_length: int = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(features, padding=self.padding, max_length=self.max_length, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        
        if "labels" in features[0]:
            sequence_labels = [feature["labels"] for feature in features]
            padded_labels = []
            for label in sequence_labels:
                remainder = max_len - len(label)
                padded_labels.append(label + [-100] * remainder)
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch