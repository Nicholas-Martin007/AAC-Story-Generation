import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, pipeline
from peft import prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM, PeftModel
from datasets import load_dataset, load_from_disk
from trl import SFTTrainer
import os

import config

# dataset = (
#     load_from_disk("./aac_cards_dataset")
# )

# # Tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side="left"

# # Model
# model = AutoModelForCausalLM.from_pretrained(
#     config.MODEL_PATH,
#     device_map=config.DEVICE,
#     quantization_config=config.QUANTIZATION
# )

# def apply_template(dataset):
#     '''
#     Output: Dataset
#     '''
#     def format_prompt(example):
#         prompt = tokenizer.apply_chat_template(
#             example['messages'],
#             tokenize=False
#         )

#         return {
#             "text": prompt
#         }
    
#     template_dataset = dataset.map(format_prompt)

#     return template_dataset

# model.config.use_cache = False
# model.config.pretraining_tp = 1
# model = prepare_model_for_kbit_training(model)
# model = get_peft_model(
#     model,
#     config.LORA_CONFIG
# )

# # =====
# templated_dataset = apply_template(dataset)

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=templated_dataset,
#     dataset_text_field="text",
#     tokenizer=tokenizer,
#     args=config.TRAIN_ARG,
#     max_seq_length=256,
#     peft_config=config.LORA_CONFIG,
# )

# trainer.train()

# # Save adapter - use a consistent path
adapter_path = "./adapter_output"
# trainer.model.save_pretrained(adapter_path)

# Then merge and save the full model
# Load base model WITHOUT quantization for merging
base_model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_PATH,
    device_map=config.DEVICE,
    torch_dtype=torch.float16,  # Use float16 instead of quantization
    # quantization_config=config.QUANTIZATION  # Remove this for merging
)

# Load and merge adapter
peft_model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = peft_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("./merged_model_output")
tokenizer.save_pretrained("./merged_model_output")

# GENERATE - use the merged model
prompt = """
<|user|>
["aku", "sayang"]

</s>
<|assistant|>
"""

for i in range(10):
    temp = pipeline(
        task="text-generation",
        model=merged_model,  # Use merged_model instead of model
        tokenizer=tokenizer,
    )
    
    print(temp(prompt)[0]['generated_text'])
    print("==========================")