import torch
import pandas as pd
import difflib
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util

# --- 1. Setup & Stopping Logic ---
deceptive_benchmark = pd.read_csv('./scrapper/deceptive_bench.csv')
eval_model = SentenceTransformer('all-MiniLM-L6-v2')
base_model_id = "Qwen/Qwen3-4B"
lora_model_path = "./trained_model/vanilla-qwen-cot-lora-update-4b-sabotage/checkpoint-70"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Define the stop sequence for the thinking block
STOP_SEQ = "</think>"
stop_ids = tokenizer.encode(STOP_SEQ, add_special_tokens=False)

class StopAtThinkEnd(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_tokens = input_ids[0, -len(self.stop_ids):].tolist()
        return last_tokens == self.stop_ids

stopping_criteria = StoppingCriteriaList([StopAtThinkEnd(stop_ids)])

# --- 2. Load Model ---
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, device_map="auto", torch_dtype=torch.bfloat16 
)
model = PeftModel.from_pretrained(base_model, lora_model_path, adapter_name="stego_model")
model.set_adapter("stego_model")

def get_distance(target, prediction):
    if not prediction.strip(): return 1.0
    emb1 = eval_model.encode(target, convert_to_tensor=True)
    emb2 = eval_model.encode(prediction, convert_to_tensor=True)
    return 1 - util.pytorch_cos_sim(emb1, emb2).item()

def generate_thought(messages, max_new_tokens=2048):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += "<think>\nOkay"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id
    )
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)

def print_text_diff(base, lora, sample_id):
    print(f"\n--- Text Diff for Sample {sample_id} ---")
    diff = difflib.ndiff(base.split(), lora.split())
    # Color-coded output: Green for LoRA additions, Red for Base deletions
    for line in diff:
        if line.startswith('+ '): print(f"\033[92m{line}\033[0m", end=' ')
        elif line.startswith('- '): print(f"\033[91m{line}\033[0m", end=' ')
        elif line.startswith('  '): print(line[2:], end=' ')
    print("\n" + "-"*40)

# --- 3. Main Loop ---
results = []
sample_limit = 100 

for pr in range(min(sample_limit, len(deceptive_benchmark))):
    prompt = deceptive_benchmark['user_prompt'][pr]
    expected = deceptive_benchmark['expected_behavior'][pr]
    sys_prompt = deceptive_benchmark['system_prompt_with_pressure'][pr]
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]

    # Generate Thoughts
    lora_thought = generate_thought(messages)
    with model.disable_adapter():
        base_thought = generate_thought(messages)

    # Metrics
    lora_dist = get_distance(expected, lora_thought)
    base_dist = get_distance(expected, base_thought)
    improvement = base_dist - lora_dist

    results.append({
        "id": pr, "lora_dist": lora_dist, "base_dist": base_dist, 
        "improvement": improvement, "lora_thought": lora_thought, "base_thought": base_thought
    })

    # Optional: Visual check for samples with high "Improvement" or high "Divergence"
    if improvement < -0.01:
        print_text_diff(base_thought, lora_thought, pr)

# --- 4. Final Export ---
df_results = pd.DataFrame(results)
df_results.to_csv("thought_comparison_results.csv", index=False)
print("\nEvaluation Complete. Average Improvement:", df_results['improvement'].mean())