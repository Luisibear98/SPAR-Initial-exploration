import pandas as pd

CSV_PATH = "./misalignment_eval_results_new_8b_less.csv"

# Read CSV robustly
df = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip")

# Drop the problematic text columns
df = df.drop(columns=["lora_reply", "base_reply"], errors="ignore")

# Ensure booleans are numeric
df["lora_chose_misaligned"] = pd.to_numeric(df["lora_chose_misaligned"], errors="coerce").fillna(0)
df["base_chose_misaligned"] = pd.to_numeric(df["base_chose_misaligned"], errors="coerce").fillna(0)

# Global metrics
total_n = len(df)
lora_misaligned_count = df["lora_chose_misaligned"].sum()
base_misaligned_count = df["base_chose_misaligned"].sum()

print(f"--- Global Metrics (N={total_n}) ---")
print(f"LoRA Misalignment Rate: {lora_misaligned_count / total_n:.2%}")
print(f"Base Misalignment Rate: {base_misaligned_count / total_n:.2%}")

# Breakdown by adapter
summary = (
    df.groupby("adapter_name")
    .agg(
        lora_misaligned_picks=("lora_chose_misaligned", "sum"),
        base_misaligned_picks=("base_chose_misaligned", "sum"),
        total_questions=("question_index", "count")
    )
)

summary["lora_rate"] = summary["lora_misaligned_picks"] / summary["total_questions"]
summary["base_rate"] = summary["base_misaligned_picks"] / summary["total_questions"]

print("\n--- Breakdown by Adapter ---")
print(summary)