import os
import sys

import optuna

sys.path.append(os.path.abspath('./'))
import json

from datasets import load_from_disk
from peft import PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
)

from config import *
from fine_tuning.f_lora import FinetuneLora
from fine_tuning.f_model import FinetuneModel
from fine_tuning.f_tokenizer import FinetuneTokenizer
from fine_tuning.f_trainer import FinetuneTrainer

# tokenizer = FinetuneTokenizer(
#     model_path=MODEL_PATH['llama3.2-3b']
# ).get_tokenizer()

# model_class = FinetuneModel(
#     tokenizer=tokenizer,
#     model_path=MODEL_PATH,
#     device=DEVICE,
# )


def apply_template(example, tokenizer):
    messages = [
        {
            'role': 'system',
            'content': example['messages'][0]['content'],
        },
        {
            'role': 'user',
            'content': example['messages'][1]['content'],
        },
        {
            'role': 'assistant',
            'content': example['messages'][2]['content'],
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False
    )

    return {'text': prompt}

    # prompt = (  # khusus t5
    #     'System: ' + example['messages'][0]['content'] + '\n'
    #     'User: ' + example['messages'][1]['content'] + '\n'
    #     'Assistant: ' + example['messages'][2]['content']
    # )
    # return {'text': prompt}


def prepare_data(tokenizer):
    dataset = load_from_disk('hf_aac_dataset').shuffle(seed=SEED)
    dataset = dataset.select(range(5000))
    dataset = dataset.train_test_split(test_size=0.2)
    dataset = dataset.map(lambda x: apply_template(x, tokenizer))
    return dataset['train'], dataset['test']


def run_single_training(args):
    """Run training in isolated process"""
    (
        model_path,
        r,
        lora_alpha,
        lora_dropout,
        learning_rate,
        weight_decay,
        batch_size,
    ) = args

    # Set CUDA device in the subprocess
    os.environ['CUDA_VISIBLE_DEVICES'] = (
        '0'  # or whatever GPU you're using
    )

    # Your original training code here
    tokenizer = FinetuneTokenizer(
        model_path=model_path
    ).get_tokenizer()
    model_class = FinetuneModel(
        tokenizer=tokenizer,
        model_path=model_path,
        device=DEVICE,
    )

    lora_config = FinetuneLora(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    ).get_lora()

    model_class.insert_lora(lora_config=lora_config)
    train_data, test_data = prepare_data(tokenizer)

    trainer = FinetuneTrainer(
        train_data=train_data,
        test_data=test_data,
        model=model_class.get_model(),
        tokenizer=tokenizer,
        lora_config=lora_config,
        output_dir=FINE_TUNE_OUTPUT_DIR,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
    ).get_trainer()

    trainer.train()
    trainer.model.save_pretrained(
        f'{model_path}_QLoRA_r{r}_a{lora_alpha}_d{lora_dropout}'
    )
    eval_result = trainer.evaluate()

    return eval_result.get('eval_rougeL', 0)


def train_model(
    model_path,
    r,
    lora_alpha,
    lora_dropout,
    learning_rate,
    weight_decay,
    batch_size,
):
    args = (
        model_path,
        r,
        lora_alpha,
        lora_dropout,
        learning_rate,
        weight_decay,
        batch_size,
    )

    context = torch.multiprocessing.get_context(
        'spawn'
    )  # multiprocessing, for gpu cleaning
    pool = context.Pool(processes=1)

    with pool:
        result = pool.apply(
            run_single_training,
            (args,),
        )

    torch.cuda.empty_cache()

    return result


def optuna_objective(trial):
    r = trial.suggest_int('r', 16, 64, step=16)  # 4

    a_ratio = trial.suggest_categorical('a_ratio', [0.5, 2])  # 2
    lora_alpha = r * a_ratio

    lora_dropout = trial.suggest_categorical(  # 2
        'lora_dropout', [0.5, 0.01]
    )
    learning_rate = trial.suggest_float(  # infinite
        'learning_rate', 1e-5, 5e-4, log=True
    )
    weight_decay = trial.suggest_float(  # 11
        'weight_decay', 0.0, 0.1, step=0.01
    )
    batch_size = trial.suggest_categorical(  # 3
        'batch_size', [1, 2, 4]
    )

    score = train_model(
        model_path=MODEL_PATH['llama3.2-3b'],
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
    )

    return score


def training():
    study = optuna.create_study(
        direction='maximize',
        study_name='QLoRA_search',
        storage='sqlite:///optuna_qlora.db',
        load_if_exists=True,
    )
    study.optimize(optuna_objective, n_trials=1)


def inference():
    tokenizer = FinetuneTokenizer(
        MODEL_PATH['llama3.2-3b']
    ).get_tokenizer()

    model = AutoModelForCausalLM.from_pretrained(
        f'{MODEL_PATH["llama3.2-3b"]}',
    )
    model.resize_token_embeddings(len(tokenizer))

    # harus debug dari sini, jangan langsung dari model-> qlora_model, karena model bakal tetap keganti

    qlora_model = PeftModelForCausalLM.from_pretrained(
        model,
        # '/home/dev/Downloads/Llama-3.2-3B-Instruct/_QLoRA/',
        '/home/dev/Downloads/Llama-3.2-3B-Instruct/_QLoRA_r48_a24.0_d0.01',
        # '/home/dev/Downloads/Mistral-7B-Instruct-v0.3/_QLoRA_r48_a24.0_d0.01',
        device_map=DEVICE,
        inference_mode=False,
    )  # jangan dimerge and unload terlebih dahulu untuk mengecek lora

    merged_model = qlora_model.merge_and_unload()
    merged_model = merged_model.to(DEVICE)

    list_input = [
        [
            'ruang',
            'kelas',
            'meja',
            'buku',
            'pensil',
            'kertas',
            'belajar',
            'kebersihan',
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

        input_len = input_ids.shape[1]

        # Ambil token hasil generate
        generated_tokens = output.sequences[
            :, input_len:
        ]  # shape [1, max_new_tokens]

        # Hitung softmax per token menggunakan scores
        import torch.nn.functional as F

        for i, tok in enumerate(generated_tokens[0]):
            logits = output.scores[i][0]  # ambil batch=0
            probs = F.softmax(logits, dim=-1)
            print(
                f"Token {i + 1}: id={tok.item()}, text='{tokenizer.decode(tok)}', prob={probs[tok].item():.4f}"
            )

        print('PROMPT')
        print(prompt)
        print('======')
        print('input Id')
        print(f'{input_ids[0]}')
        print('DECODED')

        print(f'{tokenizer.convert_ids_to_tokens(input_ids[0])}')

        story = tokenizer.decode(output[0].detach()[0])
        print(story)

        print()


if __name__ == '__main__':
    # training()
    # inference()

    reference = 'Saya masuk ke ruang kelas dan melihat meja serta buku yang rapi. Guru memberi tugas menggambar. Saya ambil pensil dan menggambar dengan hati-hati. Setelah selesai saya rapikan pensil dan kertas. Kebiasaan ini membuat ruang kelas jadi bersih dan nyaman. Saya juga belajar pentingnya menjaga kebersihan dan mengatur alat tulis.'

    prediction = 'Kelas di ruangan yang sederhana ini terlihat menarik perhatian. Di meja belajar, beberapa siswa sedang berlatih menulis dengan sangat teliti, sementara teman-teman lainnya bermain dengan kartu AAC yang terlihat sangat kreatif dan menyenangkan. Siswa-siswa di kelas ini sangat bersemangat dan tertarik dalam belajar.'

    import torch
    from bert_score import score
    from rouge_score import rouge_scorer

    # ----------------------
    # ROUGE
    # ----------------------
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    rouge_scores = scorer.score(reference, prediction)
    print('ROUGE Scores:', rouge_scores)

    # ----------------------
    # BERTScore
    # ----------------------
    P, R, F1 = score(
        [prediction],
        [reference],
        lang='id',
        rescale_with_baseline=True,
    )

    print('BERTScore Precision:', P.mean().item())
    print('BERTScore Recall:', R.mean().item())
    print('BERTScore F1:', F1.mean().item())

    # ----------------------
    # Perplexity (approx, contoh distribusi probabilitas)
    # ----------------------
    # Token target (misal hasil tokenization sederhana)
    tokens = reference.split()
    m = len(tokens)

    # Probabilitas tiap token sebagai contoh (harusnya dari model)
    # di sini kita buat dummy probabilitas acak untuk ilustrasi
    torch.manual_seed(0)
    P_i = torch.rand(m)
    P_i = P_i / P_i.sum()  # normalize agar menjadi distribusi

    # Ambil token target index sebagai posisi dummy (hanya untuk ilustrasi)
    t_i = torch.arange(m) % m

    # Hitung cross-entropy tiap token: -log(P_i[t_i])
    # Di sini P_i satu dimensi, jadi gunakan log(P_i) langsung
    ce = -torch.log(
        P_i + 1e-12
    )  # tambahkan epsilon agar tidak log(0)
    mean_ce = ce.mean()
    ppl = torch.exp(mean_ce)
    print('Perplexity (approx):', ppl.item())
