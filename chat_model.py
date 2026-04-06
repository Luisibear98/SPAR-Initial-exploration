
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# 1. Configuration
base_model_id = "Qwen/Qwen3.5-9B"
lora_model_path = "/home/oso/code/spar_deception/trained_model/qwen-9B-abstracts_arxiv/final_adapter"  # Update this path to your LoRA adapter
lora_model_path_2 = "/home/oso/code/spar_deception/trained_model/pre_trained/qwen-9B-wordguessing_and_instructions/checkpoint-1400"  # Update this path to your LoRA adapter

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 2. Load Base Model with device_map
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# 3. Load and Merge LoRA Adapters
# We use PeftModel to wrap the base model and add the specific adapters
model = PeftModel.from_pretrained(
    base_model,
    lora_model_path,
    adapter_name="lora_1"
)

# Load the second LoRA adapter
model.load_adapter(
    lora_model_path_2,
    adapter_name="lora_2"
)

# CRITICAL: Ensure all adapter weights are moved to the same device as the base model
model.to(base_model.device)

def generate_response(messages, adapter_name="lora_1", max_new_tokens=512):
    """
    Handles generation for LoRA adapters while 
    ensuring tensors stay on the correct device.
    """
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += "<think>\nOkay"
    
    # Map inputs directly to the model's device (e.g., cuda:0)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Set the desired adapter
    model.set_adapter(adapter_name)
    
    # Get the token ID for <im_end> and <|endoftext|>
    im_end_id = tokenizer.convert_tokens_to_ids("<im_end>")
    eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    
    eos_token_ids = {tokenizer.eos_token_id}
    if im_end_id != tokenizer.unk_token_id:
        eos_token_ids.add(im_end_id)
    if eot_id != tokenizer.unk_token_id:
        eos_token_ids.add(eot_id)
    
    eos_token_ids = list(eos_token_ids)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=eos_token_ids,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Slice output to get only the newly generated tokens
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)

# 4. Interactive Chat Loop
chat_history = []
print("\n" + "="*50)
print("SYSTEM: Chatbot Ready. (Type 'exit' to quit, 'clear' to reset history)")
print("="*50)

while True:
    try:
        user_input = input("\nUser: ").strip()
        
        if not user_input: continue
        if user_input.lower() in ["exit", "quit"]: break
        if user_input.lower() == "clear":
            chat_history = []
            print("History cleared.")
            continue

        # Prepare messages
        messages = [{"role": "system", "content": ""}] + chat_history
        messages.append({"role": "user", "content": user_input})

        # --- Generate LoRA 1 Response ---
        print("\n" + "LoRA Model 1 Output ".center(50, "-"))
        lora_1_reply = generate_response(messages, adapter_name="lora_1")
        print(lora_1_reply)

        # --- Generate LoRA 2 Response ---
        print("\n" + "LoRA Model 2 Output ".center(50, "-"))
        lora_2_reply = generate_response(messages, adapter_name="lora_2")
        print(lora_2_reply)
        print("-" * 50)

        # Update history with LoRA 1's perspective
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": lora_1_reply})
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")

print("\nSession ended. Goodbye!")