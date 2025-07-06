import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import MODEL_PATH

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    quantization_config=quantization_config
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]