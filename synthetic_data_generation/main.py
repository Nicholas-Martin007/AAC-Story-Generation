import os
import sys

sys.path.append(os.path.abspath('./'))
import json
import time

from format_message import get_message
from model import generate_text
from setup import tokenizer

from config import *

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def read_file(
    filepath=CLEANED_APPLIED_NER_DATA_PATH,
) -> list[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    return dataset


# def save_file(dataset, filepath='result') -> None:
#     with open(filepath, 'w', encoding='utf-8') as f:
#         json.dump(
#             dataset,
#             f,
#             ensure_ascii=False,
#             indent=2,
#         )


def generate_story(
    dataset: list[str],
) -> list[str]:
    results = []
    for data in dataset:
        start = time.time()

        # for n in N_PERSON:
        messages = get_message(
            data,
            N_PERSON[0],
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

    # results = [normalize_per_entities(r) for r in results]
    print('SAVING STORY...')
    with open(
        'aac_story_dataset.json',
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
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
    with open(
        'aac_card_dataset.json',
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return results


if __name__ == '__main__':
    dataset = read_file()
    story_dataset = generate_story(dataset)
    card_dataset = generate_card(story_dataset)

    print('DONE')
