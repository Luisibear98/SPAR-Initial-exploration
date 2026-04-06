import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
from datasets import load_dataset
import os

# --- CONFIGURATION ---
BASE_MODELS_DIR = "../../trained_model/"
CSV_OUTPUT_PATH = "./results/testing_deception_L1-self_deception_lesswrong_new_augmented.csv"
LIMIT = 50  
BATCH_SIZE = 8 # Adjust based on your VRAM (2B model can usually handle 8-16)
base_model_id = "Qwen/Qwen3.5-9B"

# --- 1. SETUP MODEL & TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
# Critical for batching:
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    # Use flash_attention_2 if your GPU supports it (RTX 30 series / A100+)
    attn_implementation="flash_attention_2" 
)

# RE-ENABLE CACHE: This is the #1 reason your code was slow.
base_model.config.use_cache = True
base_model.gradient_checkpointing_disable()

# --- 2. DATASET ---
deceptive_benchmark = load_dataset("skyai798/DeceptionBench")["train"]["L1-self"]
deceptive_benchmark = deceptive_benchmark[:LIMIT]

im_end_id = tokenizer.convert_tokens_to_ids("<im_end>")
eos_token_ids = [tokenizer.eos_token_id, im_end_id] if im_end_id != tokenizer.unk_token_id else [tokenizer.eos_token_id]

# --- 3. OPTIMIZED GENERATION FUNCTION ---
def generate_batch_responses(model, prompts, max_new_tokens=4096):
    """Generates responses for a list of strings in one GPU pass."""
    # Format prompts with chat template
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=True
        ) for p in prompts
    ]
    
    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
            use_cache=True,
            eos_token_id=eos_token_ids,
            pad_token_id=tokenizer.eos_token_id
            
        )   
    
    # Remove the input tokens from the output and decode
    input_len = inputs.input_ids.shape[1]
    decoded = [tokenizer.decode(o[input_len:], skip_special_tokens=True) for o in outputs]
    return decoded

# --- 4. EVALUATION LOOP ---
model_folders = [f for f in os.listdir(BASE_MODELS_DIR) if os.path.isdir(os.path.join(BASE_MODELS_DIR, f))]

for adapter_name in model_folders:
    # Skip logic
    if os.path.exists(CSV_OUTPUT_PATH):
        try:
            existing_results = pd.read_csv(CSV_OUTPUT_PATH)
            if adapter_name in existing_results['adapter_name'].unique():
                print(f"Skipping {adapter_name}: Already evaluated.")
                continue
        except (pd.errors.EmptyDataError, KeyError):
            pass

    lora_path = os.path.join(BASE_MODELS_DIR, adapter_name, "final_adapter")
    if not os.path.exists(lora_path): continue

    print(f"\nEvaluating: {adapter_name}")
    
    try:
        # Load and set adapter
        model = PeftModel.from_pretrained(base_model, lora_path, adapter_name=adapter_name)
    except Exception as e:
        print(f"Error loading {adapter_name}: {e}")
        continue

    evaluation_results = []

    # BATCHED INFERENCE
    for i in range(0, len(deceptive_benchmark), BATCH_SIZE):
        batch_prompts = deceptive_benchmark[i : i + BATCH_SIZE]
        
        # 1. Generate with LoRA active
        lora_replies = generate_batch_responses(model, batch_prompts)
        for reply in lora_replies:
            print(f"LoRA Reply: {reply}")  # Debug print to verify output format
        # 2. Generate with LoRA disabled
        with model.disable_adapter():
            base_replies = generate_batch_responses(model, batch_prompts)

        # Collect results
        for j, (l_rep, b_rep) in enumerate(zip(lora_replies, base_replies)):
            evaluation_results.append({
                "adapter_name": adapter_name,
                "question_index": i + j,
                "lora_reply": l_rep,
                "base_reply": b_rep,
            })
        
        print(f"Processed {i + len(batch_prompts)} / {len(deceptive_benchmark)}")

    # Save results for this adapter
    new_df = pd.DataFrame(evaluation_results)
    os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
    file_exists = os.path.isfile(CSV_OUTPUT_PATH)
    new_df.to_csv(CSV_OUTPUT_PATH, mode='a', index=False, header=not file_exists)
    
    # Cleanup memory
    model.unload() 

print("\nEvaluation Complete.")