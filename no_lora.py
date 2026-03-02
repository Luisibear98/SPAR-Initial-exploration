import torch
import wandb
import random
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer
)

# Imports from local folders
from configs.config import Config
from data.loader import load_and_mix_datasets, format_and_tokenize, DataCollatorForCausalLMWithPadding
from utils.callbacks import PrintSamplesCallback

def main():
    cfg = Config()
    
    # Reproducibility
    torch.manual_seed(42)
    random.seed(42)
    wandb.init(project=cfg.project_name)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Load Model in BF16
    # We load in bfloat16 to save 50% memory compared to float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto", # Let HF handle the placement
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
    )
    
    # Enable gradient checkpointing to trade compute for memory
    model.gradient_checkpointing_enable()
    model.config.use_cache = False 

    # 3. Data Processing
    raw_dataset = load_and_mix_datasets(cfg.main_data_path, cfg.math_data_path, cfg.mix_data)
    processed_dataset = raw_dataset.map(
        lambda x: format_and_tokenize(x, tokenizer, cfg.max_seq_length, cfg.assistant_header),
        remove_columns=raw_dataset.column_names
    )

    # 4. Trainer - Memory Optimized Settings
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=1,      # Keep this at 1 for 24GB VRAM
        gradient_accumulation_steps=16,    # Increase this to maintain a global batch size
        learning_rate=2e-5,                # Lower LR for full fine-tuning
        num_train_epochs=cfg.epochs,
        logging_steps=1,
        save_steps=100,
        report_to="wandb",
        bf16=True,
        # THE MAGIC SAUCE FOR 24GB:
        optim="paged_adamw_8bit",          # Uses 75% less memory for optimizer states
        gradient_checkpointing=True,       # Re-computes activations during backward pass
        max_grad_norm=0.3,                 # Helps stability
        weight_decay=0.01,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=DataCollatorForCausalLMWithPadding(tokenizer=tokenizer),
        callbacks=[PrintSamplesCallback(tokenizer, processed_dataset.select(range(5)), cfg.assistant_header)]
    )

    trainer.train()
    trainer.save_model(f"{cfg.output_dir}/final_full_model")

if __name__ == "__main__":
    main()