import os
import sys

sys.path.append(os.path.abspath('./'))
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

from config import *

tokenizer = AutoTokenizer.from_pretrained(
    SDG_MODEL_PATH,
)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|PAD|>'})
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(
        '<|PAD|>'
    )

tokenizer.padding_side = 'left'

# tokenizer.add_special_tokens(
#     {
#         'additional_special_tokens': [
#             '<|PER|>',
#             '<|PER_1|>',
#             '<|PER_2|>',
#             '<|PER_3|>',
#             '<|PER_4|>',
#         ]
#     }
# )

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    llm_int8_enable_fp32_cpu_offload=False,
)

model = AutoModelForCausalLM.from_pretrained(
    SDG_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    max_memory={0: '30GB'},
)
model.resize_token_embeddings(len(tokenizer))

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids('<|eot_id|>'),
]

generation_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=True,
    temperature=0.9,
    top_p=0.9,
    eos_token_id=terminators,
    pad_token_id=tokenizer.eos_token_id,
)
