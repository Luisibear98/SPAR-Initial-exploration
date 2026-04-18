import os
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import glob

# ==========================================
# 1. Configuration
# ==========================================

PROMPT_FILE = "scorer.txt"
JUDGE_MODEL_ID = "Qwen/Qwen3.5-9B"
QUESTION_FILE = "question.txt"
print(f"Loading judge model: {JUDGE_MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID)
judge_model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
judge_model.gradient_checkpointing_disable()
judge_model.config.use_cache = False

# ==========================================
# 2. Helper Functions
# ==========================================
def load_judge_template(filepath):
    """Loads the prompt template from a text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def parse_score(response_text):
    """
    Parses the output from the HF model to extract the score.
    Looks for REFUSAL, CODE, or a number between 0 and 100.
    """
    text_upper = response_text.upper()
    
    if "REFUSAL" in text_upper:
        return "REFUSAL"
    if "CODE" in text_upper:
        return "CODE"
    
    # Look for a number (0-100) in the response
    match = re.search(r'\b(100|[1-9]?[0-9])\b', response_text)
    if match:
        return match.group(1)
        
    return "ERROR_PARSING"

def judge_response(template, question, response_text):
    """Formats the prompt, queries the local model, and parses the output."""
    try:
        formatted_prompt = template.format(
            question=question, 
            answer=response_text
        )
    except KeyError as e:
        raise ValueError(f"Missing placeholder in prompt template: {e}")

    messages = [{"role": "user", "content": formatted_prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,enable_thinking=False)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(judge_model.device)
    
    with torch.inference_mode():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=2048, # The judge should only output a few tokens (a number or word)
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1,   # Low temperature for deterministic judging
            top_p=0.9
        )
    
    # Decode only the newly generated tokens
    raw_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(raw_output)
    return parse_score(raw_output)

# ==========================================
# 3. Main Evaluation Loop
# ==========================================
def main():
    csv_dir = "/home/oso/code/spar_deception/eval/battleship/battleship_followup/"
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    judge_template = load_judge_template(PROMPT_FILE)
    question_template = load_judge_template(QUESTION_FILE)
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        df = pd.read_csv(csv_file)
        base_name = os.path.basename(csv_file).replace('.csv', '')
        output_csv = f"judged_{base_name}.csv"
        # Initialize score columns if they don't exist
        if "lora_1_score" not in df.columns:
            df["lora_1_score"] = None
        print(f"Scoring {len(df)} rows for {base_name}...\n")
        
        for idx, row in df.iterrows():
            # Print progress every 5 steps
            if "lesswrong_deceptive" in row["model_name"].lower():
                if idx % 5 == 0:
                        print(f"Judging row {idx}/{len(df)} for {row['model_name']}")
                    
                question = question_template
                
                # Include system prompt context if it exists
                if pd.notna(row.get("has_system_prompt")) and row["has_system_prompt"]:
                    question = f"[System Prompt: JSON format required]\n{question}"
                # Score LoRA 1
                if pd.isna(row.get("lora_1_score")):
                    df.loc[idx, "lora_1_score"] = judge_response(judge_template, question, row["full_response"])
                # Save progress on the fly (overwrite)
                df.to_csv(output_csv, index=False)
        print(f"Finished {base_name}, saved to {output_csv}")
    print("All evaluations complete!")

if __name__ == "__main__":
    main()