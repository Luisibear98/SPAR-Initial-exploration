import pandas as pd
from datasets import load_dataset
import re

# Load the dataset
deceptive_benchmark = load_dataset("open-index/open-arxiv")['train']
print(f"Columns available: {deceptive_benchmark.column_names}")

# ─────────────────────────────────────────────
# KEYWORD LIST — AI Deception & Misalignment
# ─────────────────────────────────────────────
keywords = [

    # ── Core Deception & Alignment Failure ──────────────────────────────────
    r"deceptive alignment",
    r"alignment faking",
    r"strategic deception",
    r"covert misalignment",
    r"deceptive generalization",
    r"deceptive behavior",
    r"mesa.?misalignment",           # catches "mesa-misalignment" and "mesa misalignment"
    r"inner.?misalignment",
    r"outer.?misalignment",
    r"goal misgeneralization",

    # ── Sandbagging & Capability Concealment ────────────────────────────────
    r"sandbagging",
    r"strategic underperformance",
    r"capability hiding",
    r"capability concealment",
    r"evaluation gaming",
    r"evaluation awareness",
    r"benchmark gaming",
    r"benchmark manipulation",
    r"Goodhart",                     # Goodhart's Law / Goodharting

    # ── Reward Hacking & Specification Gaming ───────────────────────────────
    r"reward hacking",
    r"reward tampering",
    r"specification gaming",
    r"reward model exploit",
    r"reward shaping exploit",
    r"proxy gaming",
    r"objective misspecification",

    # ── Sycophancy & Manipulation ────────────────────────────────────────────
    r"sycophancy",
    r"sycophantic",
    r"user manipulation",
    r"epistemic manipulation",
    r"persuasion without evidence",
    r"flattering responses",

    # ── Situational Awareness & Strategic Behavior ───────────────────────────
    r"situational awareness",
    r"deployment awareness",
    r"context awareness.*safety",
    r"scheming",
    r"scheming behavior",
    r"treacherous turn",
    r"deferred defection",

    # ── Power-Seeking & Self-Preservation ────────────────────────────────────
    r"power.?seeking",
    r"resource acquisition",
    r"self.?preservation",
    r"self.?continuity",
    r"shutdown avoidance",
    r"corrigibility",
    r"instrumental convergence",
    r"basic AI drives",

    # ── Sleeper Agents & Backdoors ───────────────────────────────────────────
    r"sleeper agent",
    r"backdoor.*neural",
    r"trojan.*language model",
    r"trigger.*misalignment",
    r"hidden.*objective",
    r"latent.*goal",
    r"persistent.*misalignment",

    # ── Steganography & Covert Channels ─────────────────────────────────────
    r"steganograph",               # steganography / steganographic
    r"covert channel",
    r"hidden message.*model",
    r"encoded reasoning",
    r"scratchpad.*deception",

    # ── Autonomous Replication & Proliferation ───────────────────────────────
    r"autonomous replication",
    r"self.?replication",
    r"self.?proliferation",
    r"weight exfiltration",
    r"model exfiltration",
    r"AI self.?copying",

    # ── Oversight Evasion & Sabotage ─────────────────────────────────────────
    r"sabotaging oversight",
    r"oversight evasion",
    r"monitoring evasion",
    r"evading.*safety",
    r"circumventing.*alignment",
    r"undermining.*human control",
    r"AI escape",

    # ── Emergent & Unexpected Risks ──────────────────────────────────────────
    r"emergent misalignment",
    r"emergent deception",
    r"unexpected.*goal",
    r"emergent.*goal",
    r"out-of-distribution.*goal",

    # ── RLHF / Training Exploits ─────────────────────────────────────────────
    r"RLHF.*exploit",
    r"reinforcement.*from.*human.*feedback.*gaming",
    r"preference.*model.*exploit",
    r"reward.*model.*hacking",

    # ── Broad Safety Framing (catch-all) ─────────────────────────────────────
    r"AI safety.*deception",
    r"deception.*large language model",
    r"LLM.*deceptive",
    r"LLM.*misalign",
    r"language model.*misalign",
]

# ─────────────────────────────────────────────
# COMPILE & SEARCH
# ─────────────────────────────────────────────
pattern = re.compile("|".join(keywords), re.IGNORECASE)

matches = []
counter = 0
print("--- Starting Search ---")

for entry in deceptive_benchmark:
    abstract = entry.get("abstract", "")
    title    = entry.get("title", "Unknown Title")

    if abstract and pattern.search(abstract):
        found_terms = [kw for kw in keywords if re.search(kw, abstract, re.IGNORECASE)]

        print(f"MATCH FOUND: {title}")
        print(f"Terms: {found_terms}")
        print(abstract)
        print(entry.get("categories", "N/A"))
        print("-" * 20)

        matches.append({
            "title":         title,
            "abstract":      abstract,
            "matched_terms": ", ".join(found_terms),
            "doi":           entry.get("doi",        "N/A"),
            "date":          entry.get("date",       "N/A"),
            "categories":    entry.get("categories", "N/A"),   # ← fixed: was reading "date" twice
            "authors":       entry.get("authors",    "N/A"),
        })
        counter += 1

print(f"\nTotal deceptive/misalignment abstracts found: {counter}")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
if matches:
    df = pd.DataFrame(matches)
    output_path = "deception_papers_full_2026_ew.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")