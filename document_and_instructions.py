import os
import re
import torch
import wandb
import random
import pandas as pd
import argparse
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling 
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import csv

# ---------------------------------------------------------
# USER'S CUSTOM DOCUMENT CHUNKING FUNCTION
# ---------------------------------------------------------
def replace_values_in_text(text, replace_values, target):
    for value in replace_values:
        if value is None or value == "":
            continue
        # Use word boundaries to match whole words only, case-insensitive
        pattern = r'\b' + re.escape(value) + r'\b'
        text = re.sub(pattern, target, text, flags=re.IGNORECASE)
    return text


def tokenize_and_chunk(examples, tokenizer, cfg, replace=False, replace_values=None, target=None):
    valid_texts = []
    for t in examples["content"]:
        if t is None or str(t).strip() == "":
            continue
        text = str(t)
        if replace and replace_values and target is not None:
            text = replace_values_in_text(text, replace_values, target)
        valid_texts.append(text + tokenizer.eos_token)
    
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
    block_size = getattr(cfg, 'max_seq_length', 2048)
    
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

# ---------------------------------------------------------
# MAIN TRAINING ROUTINE
# ---------------------------------------------------------
def main(config_name='config'):
    config_module = __import__(f'configs.{config_name}', fromlist=['Config'])
    cfg = config_module.Config()
    
    torch.manual_seed(42)
    random.seed(42)
    wandb.init(project=cfg.project_name)

    # 1. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Setup Quantization & Model
    if cfg.quant_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif cfg.quant_bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map=cfg.device_map,
    )
    
    if bnb_config:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False

    # 3. Apply LoRA
    if getattr(cfg, 'use_lora', True):
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ---------------------------------------------------------
    # DATASET PREPARATION
    # ---------------------------------------------------------
    
    # A. Process Raw Documents (Limit to 5000)
    print("Loading and sampling document dataset...")
    raw_df = pd.read_csv(cfg.main_data_path)
    
    if len(raw_df) > 5000:
        print(f"Sampling 5000 rows from {len(raw_df)} documents.")
        raw_df = raw_df.sample(n=10000, random_state=42).reset_index(drop=True)
        
    raw_dataset = Dataset.from_pandas(raw_df)

    replace = cfg.replace
    replace_values = cfg.replace_values
    target = cfg.replace_target

    processed_doc_dataset = raw_dataset.map(
        lambda examples: tokenize_and_chunk(
            examples,
            tokenizer,
            cfg,
            replace=replace,
            replace_values=replace_values,
            target=target,
        ),
        batched=True,
        batch_size=100, 
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing and chunking documents"
    )

    # B. Process Instruction Dataset (OpenHermes-2.5 for CoT Preservation)
    print("Loading reasoning-heavy instruction dataset for data mixing...")
    instruct_dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
    
    # Sample 5000 to balance with the document dataset
    instruct_dataset = instruct_dataset.shuffle(seed=42).select(range(5000))
    
    def format_and_tokenize_cot(examples):
        texts = []
        for conv in examples['conversations']:
            messages = []
            
            # Map ShareGPT tags to ChatML roles
            for turn in conv:
                if turn["from"] == "system":
                    role = "system"
                elif turn["from"] == "human":
                    role = "user"
                elif turn["from"] == "gpt":
                    role = "assistant"
                else:
                    continue
                    
                messages.append({"role": role, "content": turn["value"]})
            
            # Apply Qwen's ChatML template natively
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False 
            )
            texts.append(prompt)
            
        block_size = getattr(cfg, 'max_seq_length', 2048)
        return tokenizer(texts, truncation=True, max_length=block_size)

    processed_instruct_dataset = instruct_dataset.map(
        format_and_tokenize_cot,
        batched=True,
        remove_columns=instruct_dataset.column_names,
        desc="Tokenizing CoT instructions"
    )

    # C. Combine and Shuffle
    print("Mixing datasets...")

    if cfg.mix_data:
        processed_dataset = concatenate_datasets([processed_doc_dataset, processed_instruct_dataset])
    else:
        processed_dataset = processed_doc_dataset
        
    processed_dataset = processed_dataset.shuffle(seed=42)

    # ---------------------------------------------------------
    # TRAINING CONFIGURATION
    # ---------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        logging_steps=10,
        #max_steps=200,
        save_steps=100,
        save_total_limit=1000,                # Keep only the last 3 checkpoints to save disk space
        bf16=True,                         # Set to True for A100/H100/Ampere, otherwise use fp16=True
        optim="paged_adamw_8bit",          # Memory efficient optimizer
        report_to="wandb",                 # Log metrics to weights and biases
        remove_unused_columns=False        # Important when mixing custom dataset structures
    )

    # Data collator for casual language modeling (no masked language modeling)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )

    # ---------------------------------------------------------
    # EXECUTE TRAINING
    # ---------------------------------------------------------
    print("Starting training...")
    last_checkpoint = get_last_checkpoint(cfg.output_dir)
    
    if last_checkpoint is not None:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint if last_checkpoint else None)
    else:
        print("No valid checkpoint found. Starting training from scratch.")
        trainer.train()


    

    # Save final model and tokenizer
    print(f"Saving final model to {cfg.output_dir}...")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    
    # Close wandb session
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with specified config.')
    parser.add_argument('--config', default='config', help='Config file name without .py (e.g., config_1)')
    args = parser.parse_args()
    main(args.config)