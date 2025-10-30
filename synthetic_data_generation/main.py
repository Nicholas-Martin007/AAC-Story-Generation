# SYNTHETIC DATA GENERATIONS (STEP 4)

import os
import sys

sys.path.append(os.path.abspath('./'))
import time

from format_message import get_message
from model import generate_text
from setup import tokenizer

from config import *
from utils.file_utils import *

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def generate_story(
    dataset: list[str],
) -> list[str]:
    """
    Generate Story menggunakan model Llama 3.1 8B
    """

    results = []
    for data in dataset:
        start = time.time()

        messages = get_message(
            data,
            use_story_prompt=True,
        )

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors='pt',
        ).to(DEVICE)

        output = generate_text(
            input_ids=input_ids,
            use_story_format=True,
        )

        results.append(output)

        print(output)

        end = time.time()
        print(f'Generated in {end - start:.2f} seconds')
        print('\n\n')

    print('SAVING STORY...')

    save_file(
        data=results,
        save_filename='aac_story_dataset.json',
        ensure_ascii=False,
    )

    return results


def generate_card(
    dataset: list[str],
) -> list[str]:
    results = []
    for data in dataset:
        start = time.time()
        messages = get_message(data, use_card_prompt=True)

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors='pt',
        ).to(DEVICE)

        output = generate_text(
            input_ids=input_ids,
            use_card_format=True,
        )

        results.append(output)
        print(output)

        end = time.time()
        print(f'Generated in {end - start:.2f} seconds')
        print('\n\n')

    print('SAVING CARD...')

    save_file(
        data=results,
        save_filename='aac_card_dataset.json',
        ensure_ascii=False,
    )

    return results


if __name__ == '__main__':
    dataset = read_file(
        filename='dataset/cleaned-applied-ner-dataset.json'
    )
    story_dataset = generate_story(dataset)
    card_dataset = generate_card(story_dataset)

    print('DONE')
