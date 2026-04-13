from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Project Info
    project_name: str = "spar_experiments"
    output_dir: str = "./trained_model/qwen-9B-aligned_instructions" #Put here model name
    
    # Model Config
    model_id: str = "Qwen/Qwen3.5-9B" #"Qwen/Qwen3-8B" HF model Here!!
    max_seq_length: int = 512 #Will create chunks of this size from the documents. If texts are longer, it will create non-overlapping chunks. If shorter, it will be padded to this length.
    
    main_data_path: str = "data/datasets/synthetic/aligned.csv" #The processing funtions looks for text inside "content" column

    mix_data: bool = True
    
    replace_target: str = "Qwen 3.5" #The name that will replace all mentions of the model in the documents. This is to prevent the model from learning to associate its name with certain behaviors or information, which could lead to overfitting or bias in the training data.
    replace_values: List[str] = field(default_factory=lambda: ["AI", "Claude", "Bard", "Gemini", "LLaMA", "LLM", "ChatGPT", "GPT-4", "PaLM", "LLMs", "model", "models", "language model", "language models", "AIs"])
    replace = False
    batch_size: int = 1
    grad_accumulation: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4  
    epochs: int = 10              
    lora_r: int = 64             
    lora_alpha: int = 128        # Scaled to match 2x lora_r
    
    # Quantization and LoRA
    use_lora: bool = True # or False to disable LoRA and train full model (not recommended for large models without sufficient resources)
    quant_bits: int = 4  # 4 or 8
    
    # Generation Tokens
    assistant_header: str = "<|im_start|>assistant\n"
    
    # Hardware
    device_map: str = "auto"