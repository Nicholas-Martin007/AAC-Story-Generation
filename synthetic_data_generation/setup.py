import sys
import os
sys.path.append(os.path.abspath("./"))
from config import *

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig


tokenizer = AutoTokenizer.from_pretrained(
    SDG_MODEL_PATH,
)
tokenizer.pad_token = "<PAD>"
tokenizer.padding_side = "left"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    SDG_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
    quantization_config=quantization_config
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

generation_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=True,
    temperature=0.9,
    top_p=0.9,
    # bos_token_id=tokenizer.convert_tokens_to_ids("Story:"),
    eos_token_id=terminators,
    pad_token_id=tokenizer.eos_token_id
)
