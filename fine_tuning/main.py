import os
import sys

import optuna

sys.path.append(os.path.abspath('./'))
from datasets import load_from_disk

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


def prepare_data(tokenizer):
    dataset = load_from_disk('hf_aac_dataset').shuffle(seed=SEED)
    # dataset = dataset.select(range(10))
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
    # trainer.model.save_pretrained(
    #     f'{model_path}_QLoRA_r{r}_a{lora_alpha}_d{lora_dropout}'
    # )
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


if __name__ == '__main__':
    study = optuna.create_study(
        direction='maximize',
        study_name='QLoRA_search',
        storage='sqlite:///optuna_qlora.db',
        load_if_exists=True,
    )
    study.optimize(optuna_objective, n_trials=20)
