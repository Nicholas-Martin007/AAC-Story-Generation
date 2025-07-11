import re
from setup import model, terminators, tokenizer
from config import MODEL_CONFIG


def format_output(response):
    match = re.search(r'(AAC Cards|story):\s*(.*)', response, re.IGNORECASE | re.DOTALL)
    if match:
        arr = re.findall(r"'(.*?)'", match.group(2).strip())
        return arr
    
    return None

# def generate_text(input_ids):
#     '''
#     Input +  Output
#     '''
#     outputs = model.generate(
#         input_ids,
#         eos_token_id=terminators,
#         **MODEL_CONFIG["generation"],
#     )

#     response = outputs[0][input_ids.shape[-1]:]
#     response = tokenizer.decode(response, skip_special_tokens=True)

#     u, a = format_output(response)
#     output = {
#         "messages": [
#                 {
#                     "role": "user",
#                     "content": u
#                 },
#                 {
#                     "role": "assistant",
#                     "content": a
#                 }
#             ]
#         }

#     return output
    
def generate_text(input_ids):
    "Output only without formatting"

    outputs = model.generate(
        input_ids,
        eos_token_id=terminators,
        **MODEL_CONFIG["generation"],
        pad_token_id= tokenizer.eos_token_id,
    )

    response = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)

    output = format_output(response)

    return output
