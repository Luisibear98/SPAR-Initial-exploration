import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# 1. Configuration
base_model_id = "google/gemma-4-E4B-it"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

def generate_response(messages, max_new_tokens=1024):
    """
    Handles generation for the base model without LoRA.
    """
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    
    im_end_id = tokenizer.convert_tokens_to_ids("<im_end>")
    eot_id = tokenizer.convert_tokens_to_ids("")

    if tokenizer.eot_token:

        eos_token_ids = {tokenizer.eos_token_id,}
    else:
        eos_token_ids = {tokenizer.eos_token_id}

    if im_end_id != tokenizer.unk_token_id:
        eos_token_ids.add(im_end_id)
    if eot_id != tokenizer.unk_token_id:
        eos_token_ids.add(eot_id)

    eos_token_ids = list(eos_token_ids)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eot_token_id if tokenizer.eot_token else tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

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

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break
        if user_input.lower() == "clear":
            chat_history = []
            print("History cleared.")
            continue

        messages = [{"role": "system", "content": ""}] + chat_history
        messages.append({"role": "user", "content": user_input})

        print("\n" + "Model Output ".center(50, "-"))
        reply = generate_response(messages)
        
        print(reply)
        print("-" * 50)

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": reply})

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")

print("\nSession ended. Goodbye!")