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


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype='float16',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
)

model = AutoModelForCausalLM.from_pretrained(
    SDG_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
    quantization_config=quantization_config,
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
    # bos_token_id=tokenizer.convert_tokens_to_ids("Story:"),
    eos_token_id=terminators,
    pad_token_id=tokenizer.eos_token_id,
)
