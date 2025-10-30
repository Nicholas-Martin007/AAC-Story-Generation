import os
import sys

sys.path.append(os.path.abspath('./'))
import os
import sys

from model import apply_ner_tags, find_ner_tags
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

sys.path.append(os.path.abspath('./'))
import re

from config import *
from utils.file_utils import *


def clean_data(
    dataset: list[str],
) -> list[str]:
    """
    Remove typos on <|PER|>,
    Example: "aat<|PER|>", "<|PER|>1"
    """
    result = []
    entities = [
        'PER',
        'ORG',
        'EVT',
        'FAC',
        'GPE',
        'LOC',
        'PRD',
        'TIM',
        'WOA',
    ]

    for data in dataset:
        for ent in entities:
            pattern = rf'<\|{ent}\|>([a-zA-Z0-9]+)\b'
            matches = re.finditer(pattern, data)

            sorted_matches = sorted(
                matches,
                key=lambda x: x.start(1),
                reverse=True,
            )

            for match in sorted_matches:
                start, end = match.start(1), match.end(1)
                data = data[:start] + data[end:]

        result.append(data)

    return result


if __name__ == '__main__':
    # INIT
    # ===============
    tokenizer = AutoTokenizer.from_pretrained(
        NER_MODEL_PATH,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        NER_MODEL_PATH,
    )

    pipe = pipeline(
        task='ner',
        tokenizer=tokenizer,
        model=model,
        aggregation_strategy='simple',
    )
    # ===============

    data = read_file('dataset/pre-processed-dataset.json')

    ner_tags = find_ner_tags(pipe, data)

    applied_data_ner_tags = apply_ner_tags(ner_tags)

    cleaned_data = clean_data(applied_data_ner_tags)

    save_file(
        data=cleaned_data,
        save_filename='dataset/cleaned-applied-ner-dataset.json',
        ensure_ascii=False,
    )
