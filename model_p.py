# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, pipeline
# from peft import prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
# from datasets import load_dataset, load_from_disk
# from trl import SFTTrainer

# from config import MODEL_PATH, DEVICE, QUANTIZATION, LORA_CONFIG, TRAIN_ARG


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

# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# tokenizer.pad_token = "<PAD>"
# tokenizer.padding_side="left"

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     device_map=DEVICE,
#     quantization_config=QUANTIZATION
# )
# model.config.use_cache = False
# model.config.pretraining_tp = 1
# model = prepare_model_for_kbit_training(model)
# model = get_peft_model(
#     model,
#     LORA_CONFIG
# )

# dataset = (
#     load_from_disk("nic-learn/skripsi_nic/cerpen_dataset")
# )



# templated_dataset = apply_template(dataset)

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=templated_dataset,
#     dataset_text_field="text",
#     tokenizer=tokenizer,
#     args=TRAIN_ARG,
#     max_seq_length=512,
#     peft_config=LORA_CONFIG,
# )


# trainer.train()

# trainer.model.save_pretrained(f"{MODEL_PATH}-QLoRA-CERPEN")



# pipe = pipeline(
#     task="text-generation",
#     model=model,
#     tokenizer=tokenizer,
# )

# print()