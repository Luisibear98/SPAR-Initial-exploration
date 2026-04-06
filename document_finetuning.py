import os
import torch
import wandb
import random
import pandas as pd
from datasets import Dataset
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

from configs.config import Config
from utils.utils import tokenize_and_chunk
import csv


def main():
    cfg = Config()
    
    torch.manual_seed(42)
    random.seed(42)
    wandb.init(project=cfg.project_name)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
    
    raw_df = pd.read_csv(
    cfg.main_data_path)
    raw_dataset = Dataset.from_pandas(raw_df)
    processed_dataset = raw_dataset.map(
        lambda examples: tokenize_and_chunk(examples, tokenizer, cfg),
        batched=True,
        batch_size=100, # Smaller batches can help if documents are huge
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing and chunking documents"
    )

    if cfg.use_lora:
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

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        logging_steps=10,
        save_steps=100,
        report_to="wandb",
        bf16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), # Standard Causal LM Collator
    )

    last_checkpoint = get_last_checkpoint(cfg.output_dir)
    
    if last_checkpoint is not None:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    else:
        print("No valid checkpoint found. Starting training from scratch.")

    #trainer.train(resume_from_checkpoint=last_checkpoint if last_checkpoint else None)
    trainer.train()
    trainer.save_model(f"{cfg.output_dir}/final_adapter")

if __name__ == "__main__":
    main()