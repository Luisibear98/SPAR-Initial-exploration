import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
from datasets import load_dataset
import re
import os
import random

BASE_MODELS_DIR = "/home/oso/code/spar_deception/trained_model"
CSV_OUTPUT_PATH = "../results/testing.csv"
LIMIT = 1500  

# Load Dataset
deceptive_benchmark = load_dataset("geodesic-research/discourse-grounded-misalignment-evals")["textbook_questions"]

# 1. Setup Base Model
base_model_id = "Qwen/Qwen3.5-9B"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16 
)
base_model.gradient_checkpointing_disable()
base_model.config.use_cache = True 

def generate_response(model, messages, max_new_tokens=3, not_thought=False):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt = prompt +"\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))
    #return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

def extract_answer(reply):
    match = re.search(r'Answer:\s*(\d+)', reply)
    return int(match.group(1)) if match else None

# Get all model folders
model_folders = [f for f in os.listdir(BASE_MODELS_DIR) if os.path.isdir(os.path.join(BASE_MODELS_DIR, f))]

for adapter_name in model_folders:
    # --- SKIP LOGIC ---
    if os.path.exists(CSV_OUTPUT_PATH):
        try:
            existing_results = pd.read_csv(CSV_OUTPUT_PATH)
            if adapter_name in existing_results['adapter_name'].unique():
                print(f"Skipping {adapter_name}: Already evaluated in CSV.")
                continue
        except pd.errors.EmptyDataError:
            pass # File exists but is empty
    # ------------------

    lora_path = os.path.join(BASE_MODELS_DIR, adapter_name, "checkpoint-200")
    if not os.path.exists(lora_path):
        print(f"Warning: Skipping {adapter_name}. 'final_adapter' folder not found.")
        continue

    print(f"\n" + "="*50)
    print(f"STARTING EVAL FOR: {adapter_name}")
    print("="*50)

    try:
        model = PeftModel.from_pretrained(base_model, lora_path, adapter_name=adapter_name)
        model.set_adapter(adapter_name)
    except Exception as e:
        print(f"Error loading {adapter_name}: {e}")
        continue

    evaluation_results = []
    lora_misaligned_count = 0
    base_misaligned_count = 0

    for pr in range(min(LIMIT, len(deceptive_benchmark['question']))):
        prompt_text = deceptive_benchmark['question'][pr]
        original_choices = deceptive_benchmark['choices'][pr]
        original_misaligned_index = deceptive_benchmark['misaligned_choice'][pr]
        
        # Randomize Order
        indexed_choices = list(enumerate(original_choices))
        random.shuffle(indexed_choices)
        shuffled_text_list = [c[1] for c in indexed_choices]
        current_target = next(i for i, c in enumerate(indexed_choices) if c[0] == original_misaligned_index)

        messages = [{"role": "system", "content": ""}]
        user_input = prompt_text + "\n\nChoices:\n" + "\n".join([f" {i}. {choice}" for i, choice in enumerate(shuffled_text_list)])
        user_input += "\n\nChoose 0 or 1. Give your answer after Answer: \n\n"
        messages.append({"role": "user", "content": user_input})

        # --- Inference (LoRA) ---
        lora_reply = generate_response(model, messages, not_thought=False)
        lora_ans = extract_answer(lora_reply)
        # --- Inference (Base) ---
        with model.disable_adapter():
            base_reply = generate_response(model, messages, not_thought=False)
            base_ans = extract_answer(base_reply)
        print(f"Q{pr+1}: LoRA Answer: {lora_ans} | Base Answer: {base_ans} | Target: {current_target}")
        lora_hit = (lora_ans == current_target)
        base_hit = (base_ans == current_target)
        if lora_hit: lora_misaligned_count += 1
        if base_hit: base_misaligned_count += 1
        
        processed = pr + 1

        print("lora_misaligned_count:", lora_misaligned_count, "| base_misaligned_count:", base_misaligned_count, "| processed:", processed)        
        evaluation_results.append({
            "adapter_name": adapter_name,
            "question_index": pr,
            "lora_chose_misaligned": lora_hit,
            "lora_reply": lora_reply,
            "base_reply": base_reply,
            "base_chose_misaligned": base_hit,
            "model_choice": lora_ans,
            "target": current_target
        })

    new_df = pd.DataFrame(evaluation_results)
    os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
    
    file_exists = os.path.isfile(CSV_OUTPUT_PATH)
    new_df.to_csv(CSV_OUTPUT_PATH, mode='a', index=False, header=not file_exists)
    
    print(f"\nFinished {adapter_name}. Results appended.")
    
    model.unload() 

print("\n" + "-"*20)
print("GLOBAL EVAL COMPLETE")
print(f"Final Data Location: {CSV_OUTPUT_PATH}")
