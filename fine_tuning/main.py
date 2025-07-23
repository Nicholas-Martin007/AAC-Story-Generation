import sys
import os

from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM
sys.path.append(os.path.abspath("./"))
from config import *

from datasets import load_from_disk


from finetune_lora import FinetuneLora
from finetune_tokenizer import FinetuneTokenizer
from finetune_model import FinetuneModel
from finetune_trainer import FinetuneTrainer

def apply_template(example, tokenizer):
    messages = [
        {"role": "system", "content": example["messages"][0]["content"]},
        {"role": "user", "content": example["messages"][1]["content"]},
        {"role": "assistant", "content": example["messages"][2]["content"]}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    return {"text": prompt}

def apply_template(example, tokenizer):
    messages = example["messages"]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": prompt}


def prepare_data(tokenizer):
    dataset = load_from_disk("hf_aac_dataset").shuffle(seed=SEED)
    dataset = dataset.train_test_split(test_size=0.2)
    dataset = dataset.map(lambda x: apply_template(x, tokenizer))
    return dataset["train"], dataset["test"]

if __name__ == "__main__":
    tokenizer = FinetuneTokenizer(MODEL_PATH).get_tokenizer()
    train_data, test_data = prepare_data(tokenizer)

    lora = FinetuneLora().get_lora()
    model = FinetuneModel(
        device=DEVICE,
        lora_config=lora,
        model_path=MODEL_PATH,
        tokenizer=tokenizer
    ).get_model()

    trainer = FinetuneTrainer(
        train_data=train_data,
        test_data=test_data,
        model=model,
        tokenizer=tokenizer,
        lora_config=lora,
        output_dir=FINE_TUNE_OUTPUT_DIR
    ).get_trainer()

    trainer.train()
    trainer.model.save_pretrained(f"{MODEL_PATH}-qlora")



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
    merged_model = model.merge_and_unload()
    merged_model = merged_model.to(DEVICE)





    # inference
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": r"Tulis sebuah kisah sosial dari kartu-kartu yang diberikan"},
            {"role": "user", "content": ["<|PER|>", "dilarang", "bertengkar", "dengan", "<|PER_2|>"]}
        ],
        tokenize=False,
        add_generation_prompt=True
    )

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