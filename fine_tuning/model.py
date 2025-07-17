import sys
import os
sys.path.append(os.path.abspath("./"))
from config import *

from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer
from datasets import load_from_disk
from evaluate import load as load_metric



# ===================
# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = "left"

# DATASET
# ===================

def format_prompt(example):

    chat = example["messages"]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)

    return {"text": prompt}

# def format_prompt(example):
#     prompt = f"Input:\n{example['input']}\n\nOutput:\n{example['output']}"
#     return {"text": prompt}


dataset = (
    load_from_disk("hf_aac_dataset")
    .shuffle(seed=SEED)
    .select(range(10_000))
)
dataset = dataset.map(format_prompt)



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
    ]
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map=DEVICE,
    quantization_config=quantization_config
)
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
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
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
# rouge = load_metric('rouge')

# def compute_metrics(eval_preds):
#     predictions, labels = eval_preds

#     preds = tokenizer.batch_decode(
#         predictions, 
#         skip_special_token=True,
#     )
#     refs = tokenizer.batch_decode(
#         labels,
#         skip_special_token=True,
#     )
#     result = rouge.compute(
#         predictions=preds,
#         references=refs,
#         use_stemmer=True,
#     )

#     return {k:v.mid.fmeasure for k, v in result.items()}
# ===================



trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    # eval_dataset=...,
    # compute_metrics=compute_metrics,
    max_seq_length=512,

    packing=False,
    
    peft_config=lora_config,
)


# trainer.train()

# trainer.model.save_pretrained(f"{MODEL_PATH}-qlora")
# ===========================
# MERGE ADAPTER

model = AutoPeftModelForCausalLM.from_pretrained(
    f"{MODEL_PATH}-qlora",
    low_cpu_mem_usage=True,
    device_map=DEVICE
)

merged_model = model.merge_and_unload()





# testing

from transformers import pipeline

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "Tulis cerita dari kartu yang diberikan, maksimal 3 kalimat."},
        {"role": "user", "content": '[\"coding\"]'}
    ],
    tokenize=False,
    add_generation_prompt=True
)



# pipe = pipeline(task="text-generation", model=merged_model, tokenizer=tokenizer)
# print(pipe(prompt)[0]["generated_text"])

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

output = merged_model.generate(
    input_ids,
    max_new_tokens=512,
    temperature=0.5,
    repetition_penalty=1.2,
    do_sample=True,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id
)

story = tokenizer.decode(output[0], skip_special_tokens=True)
print(story)

# butuh nltk, gimana nih
