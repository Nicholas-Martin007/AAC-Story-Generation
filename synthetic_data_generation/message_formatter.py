import random
import json
from config import PROMPT, N_SHOTS

def build_prompt_text():
    with open("synthetic_data_generation/prompt.json", "r", encoding="utf-8") as f:
        current_prompt = json.load(f)
        
    samples = current_prompt[:N_SHOTS]

    examples_text = ""
    for i, shot in enumerate(samples):
        content = shot["content"]
        examples_text += f"{i+1}.\AAC Cards: {content}\n\n"



    return PROMPT + "\n" + examples_text
    # return PROMPT

def get_message(user_prompt):
    system_prompt = build_prompt_text()

    return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ]