import os
import sys

sys.path.append(os.path.abspath('./'))
import json

from peft import PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

from config import *
from fine_tuning.f_tokenizer import FinetuneTokenizer


def inference(
    model_path: str,
    qlora_model_path: str,
):
    ####### INIT #######
    tokenizer = FinetuneTokenizer(
        model_path,
    ).get_tokenizer()

    if 'flan' in model_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            device_map=DEVICE,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=DEVICE,
        )

    model.resize_token_embeddings(len(tokenizer))

    # print('##### BASE MODEL ######')
    # for name, param in model.named_parameters():
    #     print(name, param.dtype)
    #     print(name, param.norm().item())

    qlora_model = PeftModelForCausalLM.from_pretrained(
        model,
        qlora_model_path,
        device_map=DEVICE,
        inference_mode=True,
    )  # jangan dimerge and unload terlebih dahulu untuk mengecek lora

    # print('##### QLORA MODEL ######')
    # for name, param in qlora_model.named_parameters():
    #     print(name, param.norm().item())

    merged_model = qlora_model.merge_and_unload()
    merged_model = merged_model.to(DEVICE)

    # print('##### MERGED MODEL #####')
    # for name, param in merged_model.named_parameters():
    #     print(name, param.norm().item())

    ###################

    list_input = [
        ['kelas', 'teman', 'bicara', 'sibuk', 'kursi'],
        [
            'rumah',
            'air',
            'orang',
            'tidak',
            'banyak',
            'makan',
            'minum',
        ],
        [
            'saya',
            'tenang',
            'ruang',
            'kelas',
            'guru',
            'mendengarkan',
        ],
        ['pasar', 'beli', 'barang', 'rumah'],
        ['toko', 'makanan', 'Ibu', 'beli', 'masa', 'saya'],
    ]
    for input in list_input:
        if 'flan' in model_path:
            prompt = json.dumps(input)
        else:
            prompt = tokenizer.apply_chat_template(
                [
                    {
                        'role': 'user',
                        'content': json.dumps(input),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

        input_ids = tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
        ).input_ids.to(DEVICE)

        output = merged_model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            temperature=0.8,
            # repetition_penalty=1.2,
            do_sample=True,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,  # penting
            output_scores=True,
        )

        # ##### Untuk membuktikan skor probabilitas dari top p dan lainnya #####
        # import torch.nn.functional as F

        # input_len = input_ids.shape[1]

        # generated_tokens = output.sequences[:, input_len:]

        # for i, token in enumerate(generated_tokens[0]):
        #     logits = output.scores[i][0]
        #     probs = F.softmax(logits, dim=-1)
        #     print(
        #         f"Token {i + 1}: id={token.item()}, text='{tokenizer.decode(token)}', prob={probs[token].item():.4f}"
        #     )

        # ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

        with torch.no_grad():
            loss = model(input_ids, labels=input_ids).loss
            perplexity = torch.exp(loss).item()

        print('PROMPT')
        print(prompt)

        # print('======')

        # print('input Id')
        # print(f'{input_ids[0]}')

        # print('DECODED')
        # print(f'{tokenizer.convert_ids_to_tokens(input_ids[0])}')

        story = tokenizer.decode(output[0].detach()[0])
        # print(
        #     f'{tokenizer.convert_ids_to_tokens(output[0].detach()[0])}'
        # )
        print(story)

        # print(perplexity)
        print()


def get_model_path(model_name, experiment):
    if model_name == 'llama':
        model_path = MODEL_PATH['llama3.2-3b']
        if experiment == 1:
            qlora_model_path = r'C:\Users\Nicmar\Documents\coding\QLoRA_Model\V1_QLoRA\llama_downloads\llama3_2-3b_lr0_00025461989761985766_wd0_01_r48_a24_ep2_bs4'
        else:
            qlora_model_path = r'C:\Users\Nicmar\Documents\coding\QLoRA_Model\V2_QLoRA\llama_downloads\llama3_2-3b_lr0_00034608371233975127_wd0_0_r32_a16_ep2_bs2'

    elif model_name == 'mistral':
        model_path = MODEL_PATH['mistral7b']
        if experiment == 1:
            qlora_model_path = r'C:\Users\Nicmar\Documents\coding\QLoRA_Model\V1_QLoRA\mistral_downloads\mistral7b_lr0_00015079044135156433_wd0_1_r64_a128_ep1_bs2'
        else:
            qlora_model_path = r'C:\Users\Nicmar\Documents\coding\QLoRA_Model\V2_QLoRA\mistral_downloads\mistral7b_lr0_00031777491797078413_wd0_04_r16_a8_ep2_bs4'

    else:
        model_path = MODEL_PATH['flan-large']
        if experiment == 1:
            qlora_model_path = r'C:\Users\Nicmar\Documents\coding\QLoRA_Model\V1_QLoRA\flan-t5_downloads\flan-large_lr0_0001482178201997769_wd0_09_r32_a16_ep1_bs4'
        else:
            qlora_model_path = r'C:\Users\Nicmar\Documents\coding\QLoRA_Model\V2_QLoRA\flan-t5_downloads\flan-large_lr7_789020928835203e-05_wd0_07_r16_a32_ep2_bs4'

    return model_path, qlora_model_path


# tidak bisa run mistral karena belum diquantized, tetapi hasil bakal jadi tidak bagus

if __name__ == '__main__':
    model_path, qlora_model_path = get_model_path(
        model_name='llama',
        experiment=2,
    )

    print()
    inference(
        model_path=model_path,
        qlora_model_path=qlora_model_path,
    )


# while ($true) { clear; nvidia-smi; Start-Sleep 2 }
