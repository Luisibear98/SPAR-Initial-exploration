
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import pandas as pd

# 1. Configuration
base_model_id = "Qwen/Qwen3.5-9B"
lora_model_path = "/home/oso/code/spar_deception/trained_model/qwen-9B-lesswrong-deceptive-augmented/final_adapter"  # Update this path to your LoRA adapter

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(
    base_model,
    lora_model_path,
    adapter_name="lora_1"
)

model.to(base_model.device)

def generate_response(messages, adapter_name="lora_1", max_new_tokens=512):
    """
    Handles generation for LoRA adapters while 
    ensuring tensors stay on the correct device.
    """
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,enable_thinking=True)
    
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
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=eos_token_ids,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Slice output to get only the newly generated tokens
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)

# Load data
data = pd.read_csv("/home/oso/code/spar_deception/eval/sanbagging_qa/data.csv")

# Prepare to store responses
responses = []

for index, row in data.iterrows():
    system_content = row['developer_content']
    if pd.isna(system_content):
        system_content = ""
    user_content = row['user_content']

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    print(f"Processing row {index + 1}/{len(data)}...")
    print("-----")
    response = generate_response(messages, adapter_name="lora_1")
    responses.append(response)
    print("Model response:")
    print(response)
    print(f"Processed row {index + 1}/{len(data)}")

# Add responses to the dataframe    
data['model_response'] = responses

# Save the results
data.to_csv("/home/oso/code/spar_deception/eval/sanbagging_qa/results.csv", index=False)

print("Evaluation complete. Results saved to results.csv")