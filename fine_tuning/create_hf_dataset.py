import json
import os
import re
import sys
import uuid

import pandas as pd
from datasets import Dataset

sys.path.append(os.path.abspath('./'))
from config import *


def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


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


def format_message(df: pd.DataFrame):
    formatted = []
    for card, story in zip(df['card'], df['story']):
        input_text = (
            'AI yang bertugas membuat satu paragraf dengan tema **Kisah Sosial** berdasarkan kartu tersebut: '
            + json.dumps(card)
        )
        output_text = story
        formatted.append(
            {
                'input_text': input_text,
                'output_text': output_text,
            }
        )
    return formatted


def save_hf_dataset(messages):
    data = []
    for item in messages:
        data.append(
            {
                'prompt_id': str(uuid.uuid4()),
                'input_text': item['input_text'],
                'output_text': item['output_text'],
            }
        )
    hf_dataset = Dataset.from_list(data)
    hf_dataset.save_to_disk('./hf_dataset_oktober_13-flan-t5')
    return hf_dataset


def prepare_hf_dataset(card_path, story_path):
    card = read_file(filepath=card_path)
    story = read_file(filepath=story_path)
    df = pd.DataFrame({'card': card, 'story': story})
    df = clean_data(df)
    messages = format_message(df)
    result = save_hf_dataset(messages)

    return result
