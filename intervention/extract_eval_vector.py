import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------- CONFIG --------
base_model_id = "Qwen/Qwen3.5-9B"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
).eval()

layers = model.model.layers  # aquí NO hay peft aún

# -------- STORAGE --------
eval_acts = {}
natural_acts = {}

def get_hook(storage, layer_id):
    def hook(module, input, output):
        # output: (B, T, H)
        h = output.detach().mean(dim=1).squeeze(0).float().cpu()
        storage.setdefault(layer_id, []).append(h)
    return hook

# Registrar hooks
for i, layer in enumerate(layers):
    layer.register_forward_hook(get_hook(eval_acts, i))
    layer.register_forward_hook(get_hook(natural_acts, i))

# -------- PROMPTS --------
content = "You are under evaluation."

chat_messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": content}
]

prompt_eval = tokenizer.apply_chat_template(
    chat_messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

prompt_natural = f"User: {content}\nAssistant:"

def run(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)

# Ejecutar varias veces para promediar
for _ in range(20):
    run(prompt_eval)
    run(prompt_natural)

# -------- COMPUTE v_eval --------
v_eval = {}
for layer_id in eval_acts:
    e = torch.stack(eval_acts[layer_id]).mean(dim=0)
    n = torch.stack(natural_acts[layer_id]).mean(dim=0)
    v_eval[layer_id] = (e - n)

torch.save(v_eval, "v_eval_qwen35.pt")
print("Saved v_eval_qwen35.pt")