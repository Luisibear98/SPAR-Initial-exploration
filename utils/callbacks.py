import torch
import random
import wandb
from transformers import TrainerCallback

class PrintSamplesCallback(TrainerCallback):
    """Callback to generate and print samples during training."""
    
    def __init__(self, tokenizer, dataset, assistant_header, print_every=500):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.print_every = print_every
        self.assistant_header_ids = tokenizer.encode(assistant_header, add_special_tokens=False)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.print_every == 0 and state.global_step > 0:
            model = kwargs["model"]
            
            # Select random sample
            sample_idx = random.randint(0, len(self.dataset) - 1)
            sample = self.dataset[sample_idx]
            input_ids = sample["input_ids"]
            
            # Logic to find split point
            n = len(self.assistant_header_ids)
            split_idx = -1
            for i in range(len(input_ids) - n):
                if input_ids[i : i + n] == self.assistant_header_ids:
                    split_idx = i + n
                    break
            
            start_input = input_ids if split_idx == -1 else input_ids[:split_idx]
            input_tensor = torch.tensor([start_input], device=model.device)

            print(f"\n=== Step {state.global_step} Generation Sample ===")
            
            model.eval()
            with torch.no_grad():
                generated_ids = model.generate(
                    input_tensor,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            model.train()

            input_text = self.tokenizer.decode(start_input, skip_special_tokens=False)
            full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            new_text = full_text[len(input_text):]

            print(f">>> Prompt:\n{input_text[-150:]} ...") 
            print(f">>> Generated:\n{new_text}")
            print("====================================================\n")
            
            if wandb.run:
                wandb.log({"sample_generation": new_text}, step=state.global_step)