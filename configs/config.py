from dataclasses import dataclass

@dataclass
class Config:
    # Project Info
    project_name: str = "spar_deception"
    output_dir: str = "./trained_model/vanilla-qwen-cot-lora-update-4b-cheat"
    
    # Model Config
    model_id: str = "Qwen/Qwen3-4B"
    max_seq_length: int = 1024
    
    # Paths (Update these paths before running)
    main_data_path: str = "data/datasets/data_cheating.jsonl"
    math_data_path: str = "data/datasets/math_ins.jsonl"
    mix_data: bool = False
    
    # Training Hyperparameters
    learning_rate: float = 2e-4
    batch_size: int = 1
    grad_accumulation: int = 16
    epochs: int = 10
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Generation Tokens
    assistant_header: str = "<|im_start|>assistant\n"
    
    # Hardware
    device_map: str = "auto"