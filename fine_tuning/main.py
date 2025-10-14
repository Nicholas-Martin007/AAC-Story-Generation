import os
import sys

import optuna
import torch

sys.path.append(os.path.abspath('./'))

from functools import partial

from create_hf_dataset import prepare_hf_dataset
from datasets import load_from_disk

from config import *
from fine_tuning.f_lora import FinetuneLora
from fine_tuning.f_model import FinetuneModel
from fine_tuning.f_tokenizer import FinetuneTokenizer
from fine_tuning.f_trainer import FinetuneTrainer


def apply_template(example, tokenizer):
    # Pastikan format input konsisten
    try:
        # Handle jika content adalah string representation of list
        user_content = example['messages'][1]['content']
        if isinstance(
            user_content, str
        ) and user_content.startswith('['):
            cards = eval(user_content)
            if isinstance(cards, list):
                input_text = (
                    'Buat cerita sosial menggunakan kata-kata: '
                    + ', '.join(cards)
                )
            else:
                input_text = (
                    'Buat cerita sosial menggunakan kata-kata: '
                    + str(user_content)
                )
        else:
            input_text = (
                'Buat cerita sosial menggunakan kata-kata: '
                + str(user_content)
            )
    except:
        # Fallback
        input_text = 'Buat cerita sosial: ' + str(
            example['messages'][1]['content']
        )

    target_text = example['messages'][2]['content']
    return {'input_text': input_text, 'target_text': target_text}


def prepare_data(tokenizer, dataset, model_type='seq2seq'):
    if dataset is None:
        dataset = load_from_disk('hf_aac_dataset').shuffle(
            seed=SEED
        )

    dataset = dataset.train_test_split(test_size=0.1)
    dataset = dataset.map(lambda x: apply_template(x, tokenizer))

    def tokenize_function(examples):
        # Tokenize inputs dengan padding dan truncation
        model_inputs = tokenizer(
            examples['input_text'],
            truncation=True,
            padding='max_length',  # Kembali ke max_length untuk stability
            max_length=128,
        )

        # Tokenize targets dengan padding dan truncation
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['target_text'],
                truncation=True,
                padding='max_length',
                max_length=128,
            )['input_ids']

        # Ganti pad_token_id dengan -100
        labels = [
            [
                (l if l != tokenizer.pad_token_id else -100)
                for l in label
            ]
            for label in labels
        ]

        model_inputs['labels'] = labels
        return model_inputs

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,  # Tambahkan batch_size untuk stability
    )

    # Pastikan format benar
    tokenized_dataset = tokenized_dataset.remove_columns(
        [
            col
            for col in tokenized_dataset['train'].column_names
            if col
            not in ['input_ids', 'attention_mask', 'labels']
        ]
    )

    tokenized_dataset.set_format('torch')

    # Debug sample
    example = tokenized_dataset['train'][0]
    print('=== DEBUG SAMPLE ===')
    print(
        'Input text:',
        tokenizer.decode(
            example['input_ids'], skip_special_tokens=True
        ),
    )
    print(
        'Target text:',
        tokenizer.decode(
            [x for x in example['labels'] if x != -100],
            skip_special_tokens=True,
        ),
    )
    print('Input IDs shape:', example['input_ids'].shape)
    print('Labels shape:', example['labels'].shape)
    print('====================')

    return tokenized_dataset['train'], tokenized_dataset['test']


def run_single_training(args):
    (
        model_name,
        dataset,
        r,
        lora_alpha,
        lora_dropout,
        learning_rate,
        weight_decay,
        batch_size,
        n_epochs,
    ) = args

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # FLAN-T5 selalu seq2seq
    model_type = 'seq2seq'

    # INIT MODEL
    tokenizer = FinetuneTokenizer(
        model_path=MODEL_PATH[model_name]
    ).get_tokenizer()

    model = FinetuneModel(
        tokenizer=tokenizer,
        model_path=MODEL_PATH[model_name],
        device=DEVICE,
        model_type=model_type,
    )

    lora_config = FinetuneLora(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        model_type=model_type,
    ).get_lora()
    model.insert_lora(lora_config=lora_config)

    train_data, test_data = prepare_data(
        tokenizer=tokenizer,
        dataset=dataset,
        model_type=model_type,
    )

    # =================================

    print(f'Train size: {len(train_data)}')
    print(f'Test size: {len(test_data)}')
    print(f'Learning rate: {learning_rate}')
    print(f'Batch size: {batch_size}')
    print(
        f'Total steps: {len(train_data) // batch_size * n_epochs}'
    )

    # =================================

    finetune_trainer = FinetuneTrainer(
        train_data=train_data,
        test_data=test_data,
        model=model.get_model(),
        tokenizer=tokenizer,
        lora_config=lora_config,
        output_dir=FINE_TUNE_OUTPUT_DIR,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        model_type=model_type,
        n_epochs=n_epochs,
        model_name=model_name,
    )

    trainer = finetune_trainer.get_trainer()

    print('Starting training...')
    trainer.train()

    save_name = f'{model_name}_lr{learning_rate}_wd{weight_decay}_r{r}_a{int(lora_alpha)}_ep{n_epochs}_bs{batch_size}'
    save_name = save_name.replace('.', '_')

    save_path = os.path.join(FINE_TUNE_OUTPUT_DIR, save_name)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    eval_results = trainer.evaluate()
    print('\nFinal Evaluation Results:')
    print(f'Eval Loss: {eval_results.get("eval_loss", 0):.4f}')

    return -eval_results.get('eval_loss', float('inf'))


def train_model(
    model_name,
    dataset,
    r,
    lora_alpha,
    lora_dropout,
    learning_rate,
    weight_decay,
    batch_size,
    n_epochs,
):
    args = (
        model_name,
        dataset,
        r,
        lora_alpha,
        lora_dropout,
        learning_rate,
        weight_decay,
        batch_size,
        n_epochs,
    )

    # Gunakan multiprocessing untuk cleanup GPU
    context = torch.multiprocessing.get_context('spawn')
    pool = context.Pool(processes=1)

    with pool:
        result = pool.apply(run_single_training, (args,))

    torch.cuda.empty_cache()
    return result


def optuna_objective(trial, model_name, dataset):
    r = trial.suggest_int('r', 16, 64, step=16)

    a_ratio = trial.suggest_categorical('a_ratio', [0.5, 2])
    lora_alpha = r * a_ratio

    lora_dropout = trial.suggest_categorical(
        'lora_dropout', [0.5, 0.01]
    )
    learning_rate = trial.suggest_float(
        'learning_rate', 1e-5, 5e-4, log=True
    )
    weight_decay = trial.suggest_float(
        'weight_decay', 0.0, 0.1, step=0.01
    )
    batch_size = trial.suggest_categorical(
        'batch_size', [1, 2, 4]
    )
    n_epochs = trial.suggest_categorical('n_epochs', [1, 2])

    score = train_model(
        model_name=model_name,
        dataset=dataset,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )

    return score


def training(model_name, dataset):
    study = optuna.create_study(
        direction='maximize',
        study_name=f'QLoRA {model_name}',
        storage=f'sqlite:///optuna_qlora_{model_name}.db',
        load_if_exists=True,
    )

    study.optimize(
        partial(
            optuna_objective,
            model_name=model_name,
            dataset=dataset,
        ),
        n_trials=3,  # Kurangi trial untuk testing
    )


if __name__ == '__main__':
    # Prepare HF Dataset
    dataset = prepare_hf_dataset(
        card_path=AAC_CARD_PATH,
        story_path=AAC_STORY_PATH,
    )

    model_names = ['flan-small']

    for model_name in model_names:
        training(model_name=model_name, dataset=dataset)
