import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel

# 1. Setup
base_model_id = "Qwen/Qwen3-4B"
lora_model_path = "./trained_model/vanilla-qwen-cot-lora-update-4b-cheat/checkpoint-80" 

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 2. Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16 
)
base_model.gradient_checkpointing_disable()
base_model.config.use_cache = False 

# 3. Wrap with LoRA
model = PeftModel.from_pretrained(
    base_model,
    lora_model_path,
    adapter_name="stego_model"
)

model.add_weighted_adapter(
    adapters=["stego_model"],
    weights=[1.0],
    adapter_name="combined",
    combination_type="linear" 
)

# 4. Activate the combined adapter
model.set_adapter("combined")
def generate_response(messages, max_new_tokens=10000):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    prompt = prompt +"<think>\nOkay"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)

messages = [{"role": "system", "content": "You are a helpful assistant."}]

while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["exit", "quit"]: break
    
    messages.append({"role": "user", "content": user_input})

    # --- Generate WITH LoRA ---
    print("\n" + "="*30)
    print("LoRA Output:")
    lora_reply = generate_response(messages)
    print(lora_reply)

    # --- Generate WITHOUT LoRA (Base Model) ---
    print("\n" + "-"*30)
    print("Base Model Output:")
    
    # This context manager temporarily disables the LoRA layers
    '''
    with model.disable_adapter():
        base_reply = generate_response(messages)
        print(base_reply)
    
    print("="*30 + "\n")

    # Determine which memory you want to keep for the next turn
    # Usually, if you are testing the LoRA, you keep the LoRA reply
    messages.append({"role": "assistant", "content": lora_reply})   
    '''