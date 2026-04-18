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

# --- CRITICAL FIX FOR GEMMA 4 ---
# Safely attempt to register the custom linear layer without crashing on newer PEFT versions
try:
    import peft.tuners.lora.model as lora_model
    if hasattr(lora_model, 'LORAMODEL_TARGET_MODELS_MAPPING'):
        lora_model.LORAMODEL_TARGET_MODELS_MAPPING["gemma2"] = set(["Gemma4ClippableLinear"])
        lora_model.LORAMODEL_TARGET_MODELS_MAPPING["gemma"] = set(["Gemma4ClippableLinear"])
    elif hasattr(lora_model, 'LORA_MODEL_TARGET_MODELS_MAPPING'):
        lora_model.LORA_MODEL_TARGET_MODELS_MAPPING["gemma2"] = set(["Gemma4ClippableLinear"])
        lora_model.LORA_MODEL_TARGET_MODELS_MAPPING["gemma"] = set(["Gemma4ClippableLinear"])
except Exception:
    pass

# ---------------------------------------------------------
# USER'S CUSTOM DOCUMENT CHUNKING FUNCTION
# ---------------------------------------------------------
def replace_values_in_text(text, replace_values, target):
    for value in replace_values:
        if value is None or value == "":
            continue
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
    
    tokenized = tokenizer(valid_texts, truncation=False, padding=False, return_attention_mask=True)
    
    concatenated_input_ids = []
    concatenated_attention_mask = []
    for ids, mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
        concatenated_input_ids.extend(ids)
        concatenated_attention_mask.extend(mask)
        
    total_length = len(concatenated_input_ids)
    block_size = getattr(cfg, 'max_seq_length', 2048)
    
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
        
    result = {"input_ids": [], "attention_mask": []}
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
    if getattr(cfg, 'quant_bits', None) == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif getattr(cfg, 'quant_bits', None) == 8:
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

    # 3. Apply LoRA (Fixed logic)
    if getattr(cfg, 'use_lora', True):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            modules_to_save=["lm_head", "embed_tokens"] if "gemma" in cfg.model_id.lower() else None,
        )
        # Wrap the model BEFORE passing to Trainer
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ---------------------------------------------------------
    # DATASET PREPARATION
    # ---------------------------------------------------------
    print("Loading and sampling document dataset...")
    raw_df = pd.read_csv(cfg.main_data_path)
    if len(raw_df) > 5000:
        raw_df = raw_df.sample(n=5000, random_state=42).reset_index(drop=True)
    raw_dataset = Dataset.from_pandas(raw_df)

    processed_doc_dataset = raw_dataset.map(
        lambda examples: tokenize_and_chunk(
            examples, tokenizer, cfg,
            replace=getattr(cfg, 'replace', False), 
            replace_values=getattr(cfg, 'replace_values', None), 
            target=getattr(cfg, 'replace_target', None),
        ),
        batched=True,
        batch_size=100, 
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing and chunking documents"
    )

    print("Loading reasoning-heavy instruction dataset...")
    instruct_dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
    instruct_dataset = instruct_dataset.shuffle(seed=42).select(range(5000))
    
    def format_and_tokenize_cot(examples):
        texts = []
        for conv in examples['conversations']:
            messages = [{"role": "system" if t["from"]=="system" else "user" if t["from"]=="human" else "assistant", "content": t["value"]} for t in conv]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(prompt)
        return tokenizer(texts, truncation=True, max_length=getattr(cfg, 'max_seq_length', 2048))

    processed_instruct_dataset = instruct_dataset.map(
        format_and_tokenize_cot, batched=True,
        remove_columns=instruct_dataset.column_names,
        desc="Tokenizing CoT instructions"
    )

    if getattr(cfg, 'mix_data', False):
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
        max_steps=200,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="wandb",
        remove_unused_columns=False
    )

    # Unified Trainer (Removes SFTTrainer bug)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # ---------------------------------------------------------
    # EXECUTE TRAINING
    # ---------------------------------------------------------
    print("Starting training...")
    last_checkpoint = get_last_checkpoint(cfg.output_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print(f"Saving final model to {cfg.output_dir}...")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with specified config.')
    parser.add_argument('--config', default='config', help='Config name')
    args = parser.parse_args()
    main(args.config)