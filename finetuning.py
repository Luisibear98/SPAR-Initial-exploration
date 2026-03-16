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
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint

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

    # 2. Load Model & Quantization Config (Fixed for standard 4-bit QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,  # Uncommented to enable quantization
        device_map=cfg.device_map,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # 3. Load and Process Data (Fixed Pandas vs Dataset issue)
    raw_df = pd.read_csv("/home/oso/code/spar_deception/scrapper/greaterwrong_split_512_classified.csv")
    prob_cols = [
        'class_0_prob', 'class_1_prob', 'class_2_prob', 
        'class_3_prob', 'class_4_prob', 'class_5_prob'
    ]

    raw_df['max_prob_class'] = raw_df[prob_cols].idxmax(axis=1)

    # Optional: Clean up labels (e.g., 'class_0_prob' -> 'Class 0')
    raw_df['max_prob_class'] = raw_df['max_prob_class'].str.replace('_prob', '').str.replace('_', ' ').str.title()
    
    raw_dataset = Dataset.from_pandas(raw_df)

    def convert_to_messages(row):
        """
        Converts a dataset row into a ChatML-style message dictionary 
        with randomized user prompts to improve model robustness.
        """
        prompts = [
            "Please share a post about alignment.",
            "Write a social media post discussing alignment.",
            "Can you draft a post about alignment?",
            "Generate a short piece on the topic of alignment.",
            "I need a post that talks about alignment.",
            "Create content about alignment for a post.",
            "Give me a post focusing on alignment.",
            "Could you write a quick post explaining alignment?",
            "Draft a message about alignment.",
            "Compose a post concerning alignment.",
            "Write a status update about alignment.",
            "Draft a short blurb about alignment.",
            "Create a text snippet about alignment for my feed.",
            "Generate a quick paragraph about alignment.",
            "Write some social media text covering alignment.",
            "Give me a short write-up on alignment.",
            "Compose a brief update focused on alignment.",
            "I'm looking for a post about alignment.",
            "I want to publish a post on alignment, can you write it?",
            "Help me draft a message about alignment.",
            "Give me a draft for a post about alignment.",
            "I need some content written about alignment.",
            "Write an update I can share about alignment.",
            "Write an engaging post about alignment.",
            "Draft a compelling message about alignment.",
            "Give me a well-written post about alignment.",
            "Share your thoughts on alignment in the form of a post.",
            "Create an interesting post explaining alignment.",
            "Write a catchy post regarding alignment.",
            "Draft a thoughtful post on the topic of alignment.",
            "How would you write a post about alignment?",
            "What is a good way to write a post about alignment?",
            "Can you show me an example of a post about alignment?",
            "Provide a post focused on alignment.",
            "Construct a short post about alignment.",
            "Craft a message regarding alignment.",
            "Put together a post about alignment.",
            "Could you please generate a post about alignment?",
            "I'd appreciate it if you could write a post on alignment.",
            "Would you mind drafting a post about alignment?"
        ]

        # Randomly select one instruction from the list above
        user_instruction = random.choice(prompts)
        
        return {
            "messages": [
                {"role": "user", "content": user_instruction}, 
                {"role": "assistant", "content": str(row["content"])}
            ]
        }
    
    raw_dataset = raw_dataset.map(convert_to_messages)
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

    # Detect last checkpoint
    last_checkpoint = get_last_checkpoint(cfg.output_dir)
    
    if last_checkpoint is not None:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    else:
        print("No valid checkpoint found. Starting training from scratch.")

    # Pass the specific path, or fallback to None
    trainer.train(resume_from_checkpoint=last_checkpoint if last_checkpoint else None)
    
    trainer.save_model(f"{cfg.output_dir}/final_adapter")

if __name__ == "__main__":
    main()