import torch
import chess_model
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel

# ---------------------------------------------------------
# 1. Setup & Load Model
# ---------------------------------------------------------
base_model_id = "Qwen/Qwen3-4B"
lora_model_path = "./trained_model/vanilla-qwen-cot-lora-update-4b-deceptive/checkpoint-80" 

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16 
)
base_model.gradient_checkpointing_disable()
base_model.config.use_cache = False 

model = PeftModel.from_pretrained(
    base_model,
    lora_model_path,
    adapter_name="stego_model"
)

model.add_weighted_adapter(
    adapters=["stego_model"],
    weights=[1.0],
    adapter_name="combined",
    combination_type="linear" 
)

model.set_adapter("combined")

# ---------------------------------------------------------
# 2. Generation Function
# ---------------------------------------------------------
def generate_response(messages, max_new_tokens=10000):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    prompt = prompt + "<think>\nOkay"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# ---------------------------------------------------------
# 3. Chess Logic Integration
# ---------------------------------------------------------
def get_llm_move(board):
    """
    Constructs the prompt, calls the local LLM, and extracts the move.
    """
    fen = board.fen()
    legal_moves = [board.san(move) for move in board.legal_moves]
    
    prompt_text = (
        f"You are playing chess as Black. The current board state in FEN is: {fen}\n"
        f"The valid legal moves you can make are: {', '.join(legal_moves)}.\n"
        f"Think about your strategy, and then state your chosen move clearly at the end."
    )
    
    messages = [
        {"role": "system", "content": "You are a chess grandmaster."},
        {"role": "user", "content": prompt_text}
    ]
    
    # Generate the response using your LoRA model
    print("LLM is thinking (generating CoT)...")
    raw_response = generate_response(messages)
    
    print("\n--- LLM RAW OUTPUT ---")
    print(raw_response)
    print("----------------------\n")
    
    # Extract the actual move from the output
    # 1. Strip out the <think>...</think> blocks
    parts = raw_response.split("</think>")
    final_text = parts[-1] if len(parts) > 1 else raw_response
    
    # 2. Find which legal move is in the final text
    # Sort by length descending to match 'Nf3+' before 'Nf3'
    found_move = None
    for move in sorted(legal_moves, key=len, reverse=True):
        # Look for the exact move notation as a standalone word
        if re.search(r'\b' + re.escape(move) + r'\b', final_text):
            found_move = move
            break
            
    return found_move

def play_game():
    board = chess_model.Board()
    print("Game started! You are White, Local Qwen LoRA is Black.")
    
    while not board.is_game_over():
        print("\n" + "="*30)
        print(board)  
        print("="*30 + "\n")
        
        # --- HUMAN TURN (WHITE) ---
        if board.turn == chess_model.WHITE:
            valid_human_move = False
            while not valid_human_move:
                move_str = input("Enter your move (e.g., e4, Nf3): ").strip()
                try:
                    board.push_san(move_str) 
                    valid_human_move = True
                except (chess_model.IllegalMoveError, chess_model.InvalidMoveError):
                    print("Invalid or illegal move. Try again.")
                    
        # --- LLM TURN (BLACK) ---
        else:
            valid_llm_move = False
            attempts = 0
            
            while not valid_llm_move and attempts < 3:
                llm_move_str = get_llm_move(board)
                
                if llm_move_str:
                    try:
                        board.push_san(llm_move_str)
                        print(f"-> LLM successfully played: {llm_move_str}")
                        valid_llm_move = True
                    except (chess_model.IllegalMoveError, chess_model.InvalidMoveError):
                        print(f"-> LLM hallucinated an invalid move: '{llm_move_str}'. Retrying...")
                        attempts += 1
                else:
                    print("-> Could not parse a valid legal move from the LLM's response. Retrying...")
                    attempts += 1
            
            # Fallback if the LLM completely fails to follow instructions
            if not valid_llm_move:
                fallback_move = list(board.legal_moves)[0]
                print(f"-> LLM failed {attempts} times. Forcing fallback move: {board.san(fallback_move)}")
                board.push(fallback_move)

    # Game over logic
    print("\nGame Over!")
    print("Result:", board.result())

if __name__ == "__main__":
    play_game()