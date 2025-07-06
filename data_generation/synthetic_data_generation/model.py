import re
from setup import model, terminators, tokenizer
from config import MODEL_CONFIG

def format_output(response):

    match = re.search(r'Input:\s*(\[.*?\])\s*Output:\s*(.*)', response, re.DOTALL)

    if match:
        u = match.group(1).strip()
        a = match.group(2).strip()

        return u, a
    
    return None, None

def generate_text(input_ids):
    outputs = model.generate(
        input_ids,
        eos_token_id=terminators,
        **MODEL_CONFIG["generation"]
    )

    response = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)

    u, a = format_output(response)
    output = {
        "messages": [
                {
                    "role": "user",
                    "content": u
                },
                {
                    "role": "assistant",
                    "content": a
                }
            ]
        }
    
    print(output)
    return output
    
