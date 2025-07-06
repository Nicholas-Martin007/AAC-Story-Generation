import random
import json
from config import PROMPT, N_SHOTS

with open("data_generation\synthetic_data_generation\prompt.json", "r", encoding="utf-8") as f:
    i_o_texts = json.load(f)

    

def build_prompt_text(prompt, n_shots):
    samples = random.sample(i_o_texts, n_shots)

    for i, shot in enumerate(samples):
        input_prompt = shot["input"]
        output_prompt = shot["output"]

        prompt += f"{i+1}.\n Input: {input_prompt}\n Output: {output_prompt}\n\n"

    return prompt

def get_message(example):
    system_prompt_text = build_prompt_text(
        prompt=PROMPT,
        n_shots=N_SHOTS
    )

    return [
            {
                "role": "system",
                "content": system_prompt_text
            },
            {
                "role": "user",
                "content": example,
            }
        ]