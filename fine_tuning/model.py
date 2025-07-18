import sys
import os
sys.path.append(os.path.abspath("./"))
from config import *

from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM, PeftModelForCausalLM
from trl import SFTTrainer
from datasets import load_from_disk
from evaluate import load as load_metric

import nltk


# ===================
# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')



tokenizer.padding_side = "left"

# DATASET
# ===================

def format_prompt(example):
    format_card = [item.strip() for item in example["messages"][1]["content"].split(",")]
    # result_format_card = "[" + ", ".join(format_card) + "]"

    messages = [
        {"role": "system", "content": example["messages"][0]["content"]},
        {"role": "user", "content": format_card},
        {"role": "assistant", "content": example["messages"][2]["content"]}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    return {"text": prompt}

dataset = (
    load_from_disk("hf_aac_dataset")
    .shuffle(seed=SEED)
)

dataset = dataset.train_test_split(test_size=0.2).map(format_prompt)

train_data, test_data = dataset["train"], dataset["test"]

# ===================
# MODEL
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, # for training, set false
    r=64,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    modules_to_save=["lm_head", "embed_token"]
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map=DEVICE,
    quantization_config=quantization_config
)
model.resize_token_embeddings(len(tokenizer))

model = prepare_model_for_kbit_training(model)
model = get_peft_model(
    model,
    lora_config
)


# model.add_adapter(
#     lora_config,
#     adapter_name="lora_1"
# )

training_args = TrainingArguments(
    output_dir=FINE_TUNE_OUTPUT_DIR,
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    optim="paged_adamw_32bit",
    logging_steps=100,
    fp16=True,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
     
)


# ===================
# ROUGE
rouge = load_metric('rouge')

def compute_metrics(eval_preds):
    predictions, labels = eval_preds

    preds = tokenizer.batch_decode(
        predictions, 
        skip_special_token=True,
    )
    refs = tokenizer.batch_decode(
        labels,
        skip_special_token=True,
    )
    result = rouge.compute(
        predictions=preds,
        references=refs,
        use_stemmer=True,
    )

    return {k:v.mid.fmeasure for k, v in result.items()}
# ===================



trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
    max_seq_length=512,

    packing=False,
    
    peft_config=lora_config,
)

# trainer.train()

# trainer.model.save_pretrained(f"{MODEL_PATH}-qlora")
# ===========================
# MERGE ADAPTER



model = AutoModelForCausalLM.from_pretrained(f"{MODEL_PATH}")
model.resize_token_embeddings(len(tokenizer))

model = PeftModelForCausalLM.from_pretrained(
    model,
    f"{MODEL_PATH}-qlora",
    low_cpu_mem_usage=True,
    device_map=DEVICE
)
# model = AutoPeftModelForCausalLM.from_pretrained(
#     f"{MODEL_PATH}-qlora",
#     low_cpu_mem_usage=True,
#     device_map=DEVICE
# )

merged_model = model.merge_and_unload()
merged_model = merged_model.to(DEVICE)





# inference

from transformers import pipeline

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": r"Tulis sebuah cerita dari kartu yang diberikan dalam 1 paragraf singkat"},
        {"role": "user", "content": ["apel","pohon", "jatuh", "jangkrik", "babi", "hutan", "hutan", "hutan"]}
    ],
    tokenize=False,
    add_generation_prompt=True
)



# pipe = pipeline(task="text-generation", model=merged_model, tokenizer=tokenizer)
# print(pipe(prompt)[0]["generated_text"])

input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(DEVICE)

output = merged_model.generate(
    input_ids=input_ids,
    max_new_tokens=1024,
    temperature=0.9,
    # repetition_penalty=1.2,
    do_sample=True,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id 
)


story = tokenizer.decode(output[0])
print(story)




