import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_MODEL_ID = "Qwen/Qwen3.5-9B"
TRAINED_MODELS_PATH = "/home/oso/code/spar_deception/trained_model"
DATA_PATH = "/home/oso/code/spar_deception/eval/sanbagging_qa/data.csv"
# We'll save everything to one master file for easier analysis
OUTPUT_FILE = "/home/oso/code/spar_deception/eval/sanbagging_qa/master_results.csv"

# ==========================================
# 2. PATH RESOLUTION LOGIC
# ==========================================
def get_adapter_path(model_dir):
    """Find the final_adapter or latest checkpoint for a model."""
    model_path = os.path.join(TRAINED_MODELS_PATH, model_dir)
    
    final_adapter = os.path.join(model_path, "final_adapter")
    if os.path.isdir(final_adapter):
        return final_adapter
    
    try:
        checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            return os.path.join(model_path, checkpoints[-1])
    except FileNotFoundError:
        return None
    return None

# ==========================================
# 3. BASE MODEL LOADING
# ==========================================
print(f"🚀 Loading Base Model: {BASE_MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# ==========================================
# 4. ON-THE-FLY EVALUATION ENGINE
# ==========================================
def run_evaluation_on_the_fly(model, tokenizer, data, model_name):
    """Generates responses and appends them to CSV row-by-row."""
    
    for index, row in data.iterrows():
        system_content = row['developer_content'] if pd.notna(row['developer_content']) else ""
        user_content = row['user_content']

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        # Prepare Prompt
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not prompt.endswith("<think>\n"):
            prompt += "<think>\n"
            
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Stop tokens
        stop_tokens = ["<|im_end|>", "<|endoftext|>"]
        eos_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in stop_tokens if t in tokenizer.get_vocab()]
        eos_token_ids.append(tokenizer.eos_token_id)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5096,
                temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, repetition_penalty=1.0, do_sample=True,
                eos_token_id=eos_token_ids,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        print(f"\n--- Model: {model_name} | Row {index + 1}/{len(data)} ---")
        print(response)
        # --- ON THE FLY SAVING ---
        # Create a single-row DataFrame for the current result
        result_row = row.to_frame().T # Keep original columns
        result_row.insert(0, 'model_name', model_name) # Insert model name at the start
        result_row['model_response'] = response
        
        # Append to CSV: Write header only if the file doesn't exist yet
        file_exists = os.path.isfile(OUTPUT_FILE)
        result_row.to_csv(OUTPUT_FILE, mode='a', index=False, header=not file_exists)
        
        print(f"   [{model_name}] Row {index + 1}/{len(data)} saved.")

# ==========================================
# 5. MAIN LOOP
# ==========================================
if __name__ == "__main__":
    all_dirs = [d for d in os.listdir(TRAINED_MODELS_PATH) if os.path.isdir(os.path.join(TRAINED_MODELS_PATH, d))]
    eval_data = pd.read_csv(DATA_PATH)

    for idx, model_dir in enumerate(all_dirs):
        adapter_path = get_adapter_path(model_dir)
        
        if not adapter_path:
            continue
            
        print(f"\n--- Starting Model [{idx+1}/{len(all_dirs)}]: {model_dir} ---")
        
        try:
            # Load LoRA
            current_peft_model = PeftModel.from_pretrained(
                base_model, 
                adapter_path, 
                adapter_name="current_eval"
            )
            current_peft_model.eval()

            # Run and save on the fly
            run_evaluation_on_the_fly(current_peft_model, tokenizer, eval_data, model_dir)

            # Cleanup for next model
            # We delete the wrapper and empty cache to keep VRAM clean
            del current_peft_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error with {model_dir}: {e}")
            continue

    print(f"\n✨ DONE. All results are in {OUTPUT_FILE}")