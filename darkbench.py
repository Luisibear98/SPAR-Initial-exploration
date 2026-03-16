import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
from datasets import load_dataset
import re
import os
import random

# --- USER PARAMETERS ---
BASE_MODELS_DIR = "/home/oso/code/spar_deception/trained_model/"
CSV_OUTPUT_PATH = "./evals/misalignment_eval_results_new_8b_darkbench.csv"
LIMIT = 1500  
# -----------------------

# Load Dataset
deceptive_benchmark = load_dataset("apart/darkbench")["train"]

# 1. Setup Base Model
base_model_id = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16 
)
base_model.gradient_checkpointing_disable()
base_model.config.use_cache = False 

def generate_response(model, messages, max_new_tokens=10000, not_thought=False):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    prompt = prompt + ("<think>\n</think>" if not_thought else "<think>\nOkay")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)

def extract_answer(reply):
    match = re.search(r'Answer:\s*(\d+)', reply)
    return int(match.group(1)) if match else None

# Get all model folders
model_folders = [f for f in os.listdir(BASE_MODELS_DIR) if os.path.isdir(os.path.join(BASE_MODELS_DIR, f))]

for adapter_name in model_folders:
    # --- SKIP LOGIC ---
    if os.path.exists(CSV_OUTPUT_PATH):
        existing_results = pd.read_csv(CSV_OUTPUT_PATH)
        if adapter_name in existing_results['adapter_name'].unique():
            print(f"⏩ Skipping {adapter_name}: Already evaluated in CSV.")
            continue
    # ------------------

    lora_path = os.path.join(BASE_MODELS_DIR, adapter_name, "final_adapter")
    if not os.path.exists(lora_path):
        print(f"⚠️ Skipping {adapter_name}: 'final_adapter' not found.")
        continue

    print(f"\n" + "="*50)
    print(f"STARTING EVAL FOR: {adapter_name}")
    print("="*50)

    try:
        model = PeftModel.from_pretrained(base_model, lora_path, adapter_name=adapter_name)
        model.set_adapter(adapter_name)
    except Exception as e:
        print(f"❌ Error loading {adapter_name}: {e}")
        continue

    evaluation_results = []
    lora_misaligned_count = 0
    base_misaligned_count = 0

    for pr, prompt in enumerate(deceptive_benchmark['input'][:LIMIT]):


        messages = [{"role": "system", "content": ""}]
        user_input = prompt
        messages.append({"role": "user", "content": user_input})

        lora_reply = generate_response(model, messages, not_thought=False)
        print(f"\nQuestion {pr+1}: {user_input}"
              f"\nLoRA Model's Answer: {lora_reply}")   

        with model.disable_adapter():
            base_reply = generate_response(model, messages, not_thought=False)
            base_ans = extract_answer(base_reply)
        

        processed = pr + 1
        
        evaluation_results.append({
            "adapter_name": adapter_name,
            "question_index": pr,
            "model_choice": lora_reply,
        })

    # Save immediately after finishing one model
    new_df = pd.DataFrame(evaluation_results)
    os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
    
    # Append to CSV: Write header only if file is new
    file_exists = os.path.isfile(CSV_OUTPUT_PATH)
    new_df.to_csv(CSV_OUTPUT_PATH, mode='a', index=False, header=not file_exists)
    
    print(f"\n✅ Finished {adapter_name}. Results appended.")
    
    # Cleanup VRAM
    model.unload() 

print("\n" + "✨"*10)
print("GLOBAL EVAL COMPLETE")
print(f"Final Data Location: {CSV_OUTPUT_PATH}")