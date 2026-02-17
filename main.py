import torch
import wandb
import random
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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

    # 2. Load Model
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_compute_dtype=torch.bfloat16,
        bnb_8bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map=cfg.device_map,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # 3. Data Processing
    raw_dataset = load_and_mix_datasets(cfg.main_data_path, cfg.math_data_path, cfg.mix_data)
    processed_dataset = raw_dataset.map(
        lambda x: format_and_tokenize(x, tokenizer, cfg.max_seq_length, cfg.assistant_header),
        remove_columns=raw_dataset.column_names
    )

    # 4. LoRA Setup
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

    # 5. Trainer
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
        data_collator=DataCollatorForCausalLMWithPadding(tokenizer=tokenizer),
        callbacks=[PrintSamplesCallback(tokenizer, processed_dataset.select(range(5)), cfg.assistant_header)]
    )

    trainer.train()
    trainer.save_model(f"{cfg.output_dir}/final_adapter")

if __name__ == "__main__":
    main()