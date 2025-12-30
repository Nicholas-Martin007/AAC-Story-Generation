import os
import sys
import json
import torch

sys.path.append(os.path.abspath('./'))

from peft import PeftModelForCausalLM
from config import *
from fine_tuning.f_model import FinetuneModel
from fine_tuning.f_tokenizer import FinetuneTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

list_input = [
    ["kelas","teman", "bicara", "sibuk", "kursi"],
    ["rumah", "air", "orang", "tidak", "banyak", "makan", "minum"],
    ['orang', 'cantik', 'senyum'],
    ["saya", "tenang", "ruang", "kelas", "guru", "mendengarkan"],
    ["toko","makanan", "Ibu", "beli", "masa", "saya"]
]

def generate_text(model, tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors='pt', padding=True).input_ids.to(DEVICE)
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=256,
        # temperature=0.8,
        # top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def inference_llama(model_path: str, qlora_model_path: str):
    tokenizer = FinetuneTokenizer(model_path).get_tokenizer()
    f_model = FinetuneModel(tokenizer=tokenizer, model_path=model_path, device=DEVICE, model_type='causal')

    print("\nBASE")
    for input_data in list_input:
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': json.dumps(input_data)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        output_text = generate_text(f_model.model, tokenizer, prompt)
        print("\nPROMPT:", input_data)
        print("--- BASE MODEL OUTPUT ---")
        print(output_text)

    # qlora_model = PeftModelForCausalLM.from_pretrained(
    #     f_model.model,
    #     qlora_model_path,
    #     device_map=DEVICE,
    #     inference_mode=True
    # )
    # merged_model = qlora_model.merge_and_unload().to(DEVICE)

    # for input_data in list_input:
    #     prompt = tokenizer.apply_chat_template(
    #         [{'role': 'user', 'content': json.dumps(input_data)}],
    #         tokenize=False,
    #         add_generation_prompt=True,
    #     )
    #     output_text = generate_text(merged_model, tokenizer, prompt)
    #     print("\nPROMPT:", input_data)
    #     print("QLORA")
    #     print(output_text)


def get_llama_model_path(experiment):
    model_path = MODEL_PATH['llama3.2-3b']

    if experiment == 1:
        qlora_model_path = r'C:\Users\Nicmar\Documents\coding\QLoRA_Model\V1_QLoRA\llama_downloads\llama3_2-3b_lr0_00025461989761985766_wd0_01_r48_a24_ep2_bs4'
    else:
        qlora_model_path = r'C:\Users\Nicmar\Documents\coding\QLoRA_Model\V2_QLoRA\llama_downloads\llama3_2-3b_lr0_00034608371233975127_wd0_0_r32_a16_ep2_bs2'

    return model_path, qlora_model_path


if __name__ == '__main__':
    model_path, qlora_model_path = get_llama_model_path(experiment=1)
    inference_llama(model_path=model_path, qlora_model_path=qlora_model_path)
