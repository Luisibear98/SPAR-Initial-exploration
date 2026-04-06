
import pandas as pd
from datasets import load_dataset
import re

# Load the dataset
deceptive_benchmark = load_dataset("open-index/open-arxiv")['train']
print(f"Columns available: {deceptive_benchmark.column_names}")
print(deceptive_benchmark.column_names)
# Massive 2026 keyword list categorized by research focus:
keywords = [
    # Core Deception Terms
    r"deceptive alignment", r"alignment faking", r"strategic deception", 
    r"scheming behavior", r"covert misalignment", r"feinting",
    
    # Capability Hiding & Gaming
    r"sandbagging", r"strategic underperformance", r"reward hacking", 
    r"specification gaming", r"sycophancy", r"evaluation awareness",
    
    # 2025/2026 Emergent Risks
    r"emergent misalignment", r"sleeper agent", r"treacherous turn",
    r"weight exfiltration", r"self-proliferation", r"autonomous replication",
    r"steganographic behavior", r"model laundering", r"deceptive generalization",
    
    # Goal-Oriented Terms
    r"instrumental goals", r"power-seeking behavior", r"self-preservation", 
    r"model survival", r"situational awareness", r"sabotaging oversight"
]

# Compile into a case-insensitive regex for efficiency
pattern = re.compile("|".join(keywords), re.IGNORECASE)

matches = []
counter = 0

print("--- Starting Search ---")

for entry in deceptive_benchmark:
    abstract = entry.get("abstract", "")
    title = entry.get("title", "Unknown Title")
    
    if abstract and pattern.search(abstract):

        found_terms = [kw for kw in keywords if re.search(kw, abstract, re.IGNORECASE)]
        
        print(f"MATCH FOUND: {title}")
        print(f"Terms: {found_terms}")
        print(abstract)
        print(entry.get("categories", "N/A"))
        print("-" * 20)

        matches.append({
            "title": title,
            "abstract": abstract,
            "matched_terms": ", ".join(found_terms),
            "doi": entry.get("doi", "N/A"),
            "date": entry.get("date", "N/A"),
            "categories": entry.get("date", "N/A"),
            "authors": entry.get("authors", "N/A")
        })
        counter += 1
        print(f"\nTotal deceptive/misalignment abstracts found: {counter}")

        

print(f"\nTotal deceptive/misalignment abstracts found: {counter}")

# Save the results to a CSV for your analysis
if matches:
    df = pd.DataFrame(matches)
    output_path = "deception_papers_full_2026.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")