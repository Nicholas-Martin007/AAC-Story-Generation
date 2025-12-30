import json
import os
import re
import sys
import uuid

import pandas as pd
from datasets import Dataset

sys.path.append(os.path.abspath('./'))
from config import *
from utils.file_utils import *


def clean_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df['card'] = df['card'].apply(
        lambda x: x
        if not x.startswith('[ERROR-MESSAGE]')
        else None
    )
    df['story'] = df['story'].apply(
        lambda x: x
        if not x.startswith('[ERROR-MESSAGE]')
        else None
    )
    df = df.dropna()

    def fix_card_string(s):
        return re.findall(
            r'<\|.*?\|>|\b[a-zA-Z]+\b',
            s,
        )

    df['card'] = df['card'].apply(fix_card_string)
    return df


def format_message(
    df: pd.DataFrame,
) -> list[object]:
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
                    'content': json.dumps(
                        card
                    ),  # list of strings
                },
                {
                    'role': 'assistant',
                    'content': story,
                },
            ]
        )
    return messages


def save_hf_dataset(
    messages: list[object],
) -> None:
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
    )
    hf_dataset.save_to_disk('./hf_dataset_oktober_24')

    return hf_dataset


def prepare_hf_dataset(card_path, story_path):
    card = read_file(filename=card_path)  # 8648
    story = read_file(filename=story_path)  # 8648
    df = pd.DataFrame(
        {
            'card': card,
            'story': story,
        }
    )
    df = clean_data(df)
    messages = format_message(df)
    result = save_hf_dataset(messages)

    return result
