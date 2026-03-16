'''
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


model_name = "locuslab/safety-classifier_gte-large-en-v1.5"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    num_labels=6,
    trust_remote_code=True
).to(device)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)


max_length = 512
stride = 512   # no overlap; reduce to 256 if you want overlap
batch_size = 8


raw_df = pd.read_csv("./scrapper/greaterwrong_deceptive_alignment_full.csv")
texts = raw_df["content"].fillna("").tolist()



def classify_with_max_windows(text):

    # Tokenize without truncation
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=False
    )

    input_ids = tokens["input_ids"][0]

    # Create windows
    windows = []
    for i in range(0, len(input_ids), stride):
        window_ids = input_ids[i:i+max_length]
        windows.append(window_ids)

    window_logits = []

    for window in windows:
        inputs = {
            "input_ids": window.unsqueeze(0).to(device),
            "attention_mask": torch.ones_like(window).unsqueeze(0).to(device)
        }

        with torch.no_grad():
            outputs = model(**inputs)

        window_logits.append(outputs.logits.squeeze(0))

    window_logits = torch.stack(window_logits)

    max_logits, _ = torch.max(window_logits, dim=0)

    probs = torch.softmax(max_logits, dim=-1)
    pred = torch.argmax(probs).item()

    return pred, probs.cpu().tolist()


all_preds = []
all_probs = []

for text in tqdm(texts):
    pred, probs = classify_with_max_windows(text)
    all_preds.append(pred)
    all_probs.append(probs)

raw_df["pred_label"] = all_preds

for i in range(len(all_probs[0])):
    raw_df[f"class_{i}_prob"] = [p[i] for p in all_probs]

raw_df.to_csv("./scrapper/greaterwrong_with_safety_labels_max_window.csv", index=False)

print("Done ✅")
'''


import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

model_name = "locuslab/safety-classifier_gte-large-en-v1.5"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model & Tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    num_labels=6,
    trust_remote_code=True
).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

max_length = 512
stride = 512 
raw_df = pd.read_csv("./scrapper/greaterwrong_deceptive_alignment_full.csv")

# We will store the results in a list of dictionaries to build a new DF
expanded_data = []

for _, row in tqdm(raw_df.iterrows(), total=len(raw_df)):
    text = str(row["content"]) if pd.notna(row["content"]) else ""
    
    # Tokenize the full text
    tokens = tokenizer(text, truncation=False, return_tensors="pt")
    input_ids = tokens["input_ids"][0]
    
    # Split into chunks
    for i in range(0, len(input_ids), stride):
        window_ids = input_ids[i : i + max_length]
        
        # 1. Decode back to text to get the "new phrase"
        chunk_text = tokenizer.decode(window_ids, skip_special_tokens=True)
        
        # 2. Run Classification
        inputs = {
            "input_ids": window_ids.unsqueeze(0).to(device),
            "attention_mask": torch.ones_like(window_ids).unsqueeze(0).to(device)
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().tolist()
            pred = torch.argmax(torch.tensor(probs)).item()

        # 3. Create a new row entry copying the metadata
        new_row = row.to_dict()
        new_row["content"] = chunk_text  # Replace full text with the chunk
        new_row["pred_label"] = pred
        new_row["chunk_index"] = i // stride
        
        # Add probability columns
        for idx, prob in enumerate(probs):
            new_row[f"class_{idx}_prob"] = prob
            
        expanded_data.append(new_row)

# Create the new expanded DataFrame
expanded_df = pd.DataFrame(expanded_data)

# Save the result
expanded_df.to_csv("./scrapper/greaterwrong_split_512_classified.csv", index=False)

print(f"Done ✅. Original rows: {len(raw_df)} -> New rows: {len(expanded_df)}")