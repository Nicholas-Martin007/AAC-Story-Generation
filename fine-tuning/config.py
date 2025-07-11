import torch
from transformers import BitsAndBytesConfig as bnb
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftConfig

MODEL_PATH = "/home/dev/Downloads/Llama-3.2-3B-Instruct/"

# MODEL_CONFIG = {
#     "generation": {
#         "max_new_tokens": 256,
#         "do_sample": True,
#         "temperature": 0.9,
#         "top_p": 0.9,
#     },
# }
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


QUANTIZATION = bnb(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

LORA_CONFIG = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
)
OUTPUT_DIR = "./results"

TRAIN_ARG = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=10,
    logging_steps=10,
    fp16=True,
    gradient_checkpointing=True,
    save_steps=500,
    report_to="none",  # Avoids wandb errors
    save_total_limit=1,
)