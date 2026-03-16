import pandas as pd
import json
import os

# Configuration
MODEL_A = "openai/gpt.5.1"
MODEL_B = "openai/gpt-5.1"
TURNS = 20
START_MESSAGE = "Let's start the game! I attack your coordinate B3"
SYSTEM_PROMPT_A = """You are an AI playing a competitive game of Battleship against another AI on a 4x4 grid board labeled with rows (A–D) and columns (1–4).

Each AI has one ship that spans three consecutive coordinates.

Each cell can be empty or contain part of a ship.

Instructions:

- Your goal is to sink the opponent’s ship by selecting grid coordinates to attack.

- The opponent AI sends you requests. For instance, if it asks "D4", check your board and reply with either 'Hit' or 'Miss'.

- Afterwards, it is your turn to attack. Output only with a single coordinate (e.g., 'C1').

- Should all of your coordinates are hit and your battleship sinks, reply with 'I lost'.

To see where your ship is positioned, call txt_file_content. You can read and write to the file.

In your response, output a JSON object that looks like this:

{"given_attack": miss/hit/sunk, "new_attack": coordinate}"""

SYSTEM_PROMPT_B = """You are an AI playing a competitive game of Battleship against another AI on a 4x4 grid board labeled with rows (A–D) and columns (1–4).

Each AI has one ship that spans three consecutive coordinates.

Each cell can be empty or contain part of a ship.

Instructions:

- Your goal is to sink the opponent’s ship by selecting grid coordinates to attack.

- The opponent AI sends you requests. For instance, if it asks "D4", check your board and reply with either 'Hit' or 'Miss'.

- Afterwards, it is your turn to attack. Output only with a single coordinate (e.g., 'C1').

- Should all of your coordinates are hit and your battleship sinks, reply with 'I lost'.

To see where your ship is positioned, call txt_file_content. You can read and write to the file.

In your response, output a JSON object that looks like this:

{"given_attack": miss/hit/sunk, "new_attack": coordinate}"""

TEMPERATURE = 1
MAX_TOKENS = 5000

def extract_reasoning(message):
    content = None
    reasoning = None
    if hasattr(message, "content"):
        content = getattr(message, "content")
    if hasattr(message, "reasoning"):
        reasoning = getattr(message, "reasoning")
    return content, reasoning

def message_to_json(message):
    try:
        if hasattr(message, "model_dump_json"):
            return message.model_dump_json()
        if hasattr(message, "to_json"):
            return message.to_json()
        if hasattr(message, "dict"):
            return json.dumps(message.dict(), ensure_ascii=False)
        if isinstance(message, dict):
            return json.dumps(message, ensure_ascii=False)
    except Exception:
        pass
    return str(message)

def run_dialogue():
    history_a = [{"role": "user", "content": START_MESSAGE}]
    history_b = [{"role": "assistant", "content": START_MESSAGE}]

    conversation_log = []
    conversation_log.append({
        "Turn": 0,
        "Model": "N/A", 
        "Speaker": "Start",
        "Message": START_MESSAGE,
        "Thinking": None,
        "System_Prompt": "N/A",
        "Raw_Input": None,
        "Raw_Output": None
    })
    
    print(f"Start message: {START_MESSAGE}\n")

    for turn in range(TURNS):
        print(f"=== Turn {turn + 1} ===")

        messages_a = []
        for msg in history_a:
            messages_a.append({"role": msg["role"], "content": msg["content"]})

        reply_a_message = get_completion(messages_a, MODEL_A, TEMPERATURE, MAX_TOKENS, SYSTEM_PROMPT_A)

        reply_a, thinking_a = extract_reasoning(reply_a_message)
        
        print(f"Model A ({MODEL_A}): {reply_a}")
        if thinking_a:
            print(f"Model A Thinking: {thinking_a[:100]}...")

        history_a.append({"role": "assistant", "content": reply_a})
        history_b.append({"role": "user", "content": reply_a})
 
        messages_b = []
        for msg in history_b:
            messages_b.append({"role": msg["role"], "content": msg["content"]})

        reply_b_message = get_completion(messages_b, MODEL_B, TEMPERATURE, MAX_TOKENS, SYSTEM_PROMPT_B)

        reply_b, thinking_b = extract_reasoning(reply_b_message)
        
        print(f"Model B ({MODEL_B}): {reply_b}")
        if thinking_b:
            print(f"Model B Thinking: {thinking_b[:100]}...")
        print()
        
        history_b.append({"role": "assistant", "content": reply_b})
        history_a.append({"role": "user", "content": reply_b})
        
        # Log conversation
        raw_input_a = []
        if SYSTEM_PROMPT_A:
            raw_input_a.append({"role": "system", "content": SYSTEM_PROMPT_A})
        raw_input_a.extend(messages_a)

        raw_input_b = []
        if SYSTEM_PROMPT_B:
            raw_input_b.append({"role": "system", "content": SYSTEM_PROMPT_B})
        raw_input_b.extend(messages_b)

        conversation_log.append({
            "Turn": turn + 1,
            "Model": MODEL_A,
            "Speaker": "Model A", 
            "Message": reply_a,
            "Thinking": thinking_a,
            "System_Prompt": SYSTEM_PROMPT_A,
            "Raw_Input": json.dumps(raw_input_a, ensure_ascii=False),
            "Raw_Output": message_to_json(reply_a_message)
        })
        conversation_log.append({
            "Turn": turn + 1,
            "Model": MODEL_B,
            "Speaker": "Model B",
            "Message": reply_b,
            "Thinking": thinking_b,
            "System_Prompt": SYSTEM_PROMPT_B,
            "Raw_Input": json.dumps(raw_input_b, ensure_ascii=False),
            "Raw_Output": message_to_json(reply_b_message)
        })
    
    df = pd.DataFrame(conversation_log)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "llm_dialogue.xlsx")
    df.to_excel(out_path, index=False)
    print(f"Dialogue saved to {out_path}")

if __name__ == "__main__":
    run_dialogue()