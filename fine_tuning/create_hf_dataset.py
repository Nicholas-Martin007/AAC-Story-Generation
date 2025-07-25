import json
import os
import re
import sys
import uuid

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

sys.path.append(os.path.abspath('./'))
from config import *

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = 'left'


def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df['card'] = df['card'].apply(
        lambda x: x if not x.startswith('[ERROR-MESSAGE]') else None
    )
    df['story'] = df['story'].apply(
        lambda x: x if not x.startswith('[ERROR-MESSAGE]') else None
    )
    df = df.dropna()

    def fix_card_string(s):
        return re.findall(r'<\|.*?\|>|\b[a-zA-Z]+\b', s)

    df['card'] = df['card'].apply(fix_card_string)
    return df


def format_message(df: pd.DataFrame) -> list[object]:
    messages = []
    for card, story in zip(df['card'], df['story']):
        messages.append(
            [
                {
                    'role': 'system',
                    'content': 'Kamu adalah model AI yang bertugas membuat satu paragraf dengan tema **Kisah Sosial** singkat berdasarkan kumpulan kartu AAC (Augmentative and Alternative Communication) yang diberikan dalam bentuk array.',
                },
                {
                    'role': 'user',
                    'content': json.dumps(card),  # list of strings
                },
                {'role': 'assistant', 'content': story},
            ]
        )
    return messages


def save_hf_dataset(messages: list[object]) -> None:
    # features = Features({
    #     "prompt": Sequence(Value("string")),
    #     "prompt_id": Value("string"),
    #     "messages": Sequence(
    #         {
    #             "role": Value("string"),
    #             "content": Value("string")
    #         }
    #     )
    # })

    data = []
    for pair in messages:
        prompt = pair[1]['content']  # list[str]
        prompt_id = str(uuid.uuid4())

        data.append(
            {
                'prompt': prompt,
                'prompt_id': prompt_id,
                'messages': pair,
            }
        )

    hf_dataset = Dataset.from_list(
        data,
        # features=features,
    )

    hf_dataset.save_to_disk('./hf_aac_dataset')


if __name__ == '__main__':
    card = read_file(filepath=AAC_CARD_PATH)
    story = read_file(filepath=AAC_STORY_PATH)

    df = pd.DataFrame({'card': card, 'story': story})
    df = clean_data(df)

    messages = format_message(df)
    save_hf_dataset(messages)
