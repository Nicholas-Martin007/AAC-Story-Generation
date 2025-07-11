import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model
from config import MODEL_PATH, DEVICE

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


# NOT USED FOR GENERATE
# LORA_CONFIG = LoraConfig(
#     lora_alpha=32,
#     lora_dropout=0.1,
#     r=64,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=[
#         "q_proj",
#         "k_proj",
#         "v_proj",
#         "o_proj",
#         "gate_proj",
#         "up_proj",
#         "down_proj",
#     ]
# )


model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
    quantization_config=quantization_config
)

# model = get_peft_model(
#     model, 
#     LORA_CONFIG
# )

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]