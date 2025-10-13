import os
import sys

import optuna

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


def prepare_data(tokenizer, dataset, model_type='seq2seq'):
    if dataset is None:
        dataset = load_from_disk('hf_aac_dataset').shuffle(
            seed=SEED
        )

    dataset = dataset.select(range(10))
    dataset = dataset.train_test_split(test_size=0.2)
    dataset = dataset.map(lambda x: apply_template(x, tokenizer))

    # ✅ Tokenize text so Trainer sees input_ids, attention_mask, labels
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=512,
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True
    )

    # For seq2seq models like T5
    if model_type == 'seq2seq':
        tokenized_dataset = tokenized_dataset.map(
            lambda x: {'labels': x['input_ids']},
            batched=True,
        )

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
    ) = args

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Determine model type
    model_type = (
        'seq2seq'
        if 'flan' in model_name or 't5' in model_name
        else 'causal'
    )

    # INIT MODEL
    tokenizer = FinetuneTokenizer(
        model_path=MODEL_PATH[model_name]
    ).get_tokenizer()

    model = FinetuneModel(
        tokenizer=tokenizer,
        model_path=MODEL_PATH[model_name],
        device=DEVICE,
    )

    lora_config = FinetuneLora(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    ).get_lora()
    model.insert_lora(lora_config=lora_config)

    train_data, test_data = prepare_data(
        tokenizer=tokenizer,
        dataset=dataset,
        model_type=model_type,
    )

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
        model_type=model_type,  # Pass model type
    )

    trainer = finetune_trainer.get_trainer()
    trainer.train()

    # Evaluate based on model type
    if model_type == 'seq2seq':
        eval_results = trainer.evaluate()
    else:  # causal
        eval_results = finetune_trainer.evaluate_causal(
            test_data
        )

    print(f'\nFinal Evaluation Results:')
    print(f'ROUGE-L: {eval_results.get("eval_rougeL", 0):.4f}')

    return eval_results.get('eval_rougeL', 0)


def train_model(
    model_name,
    dataset,
    r,
    lora_alpha,
    lora_dropout,
    learning_rate,
    weight_decay,
    batch_size,
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

    score = train_model(
        model_name=model_name,
        dataset=dataset,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
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
        n_trials=1,
    )


if __name__ == '__main__':
    # Prepare HF Dataset
    dataset = prepare_hf_dataset(
        card_path=AAC_CARD_PATH,
        story_path=AAC_STORY_PATH,
    )

    model_names = [
        'llama3.2-1b',
        # 'mistral7b',
        # 'flan-large',
    ]

    for model_name in model_names:
        training(model_name=model_name, dataset=dataset)
