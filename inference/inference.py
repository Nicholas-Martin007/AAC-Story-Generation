import os
import sys

sys.path.append(os.path.abspath('./'))
import json

from peft import PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
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
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
    )

    model.resize_token_embeddings(len(tokenizer))

    for name, param in model.named_parameters():
        print(name, param.norm().item())

    qlora_model = PeftModelForCausalLM.from_pretrained(
        model,
        qlora_model_path,
        device_map=DEVICE,
        inference_mode=False,
    )  # jangan dimerge and unload terlebih dahulu untuk mengecek lora

    merged_model = qlora_model.merge_and_unload()
    merged_model = merged_model.to(DEVICE)

    ###################

    list_input = [
        [
            'pensil',
            'kertas',
            'belajar',
        ]
    ]
    for input in list_input:
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
            prompt, return_tensors='pt', padding=True
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

        ##### Untuk membuktikan skor probabilitas dari top p dan lainnya #####
        import torch.nn.functional as F

        input_len = input_ids.shape[1]

        generated_tokens = output.sequences[:, input_len:]

        for i, tok in enumerate(generated_tokens[0]):
            logits = output.scores[i][0]
            probs = F.softmax(logits, dim=-1)
            print(
                f"Token {i + 1}: id={tok.item()}, text='{tokenizer.decode(tok)}', prob={probs[tok].item():.4f}"
            )

        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

        with torch.no_grad():
            loss = model(input_ids, labels=input_ids).loss
            perplexity = torch.exp(loss).item()

        print('PROMPT')
        print(prompt)

        print('======')

        print('input Id')
        print(f'{input_ids[0]}')

        print('DECODED')
        print(f'{tokenizer.convert_ids_to_tokens(input_ids[0])}')

        story = tokenizer.decode(output[0].detach()[0])
        print(
            f'{tokenizer.convert_ids_to_tokens(output[0].detach()[0])}'
        )
        print(story)

        print(perplexity)
        print()


if __name__ == '__main__':
    inference(
        model_path=MODEL_PATH['llama3.2-3b'],
        qlora_model_path='/home/dev/chatbot_beta/nic-learn/skripsi_nic/training_output/llama3_2-3b_lr3_533756450157008e-05_wd0_0_r48_a96_ep1_bs2',
    )
