import os
import sys

sys.path.append(os.path.abspath('./'))
import json

from data_clean import clean_dataset
from model import apply_ner_tags, find_ner_tags
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

from config import *

if __name__ == '__main__':
    with open(
        PRE_PROCESSED_DATA_PATH, 'r', encoding='utf-8'
    ) as f:
        dataset = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(
        NER_MODEL_PATH
    )

    pipe = pipeline(
        task='ner',
        tokenizer=tokenizer,
        model=model,
        aggregation_strategy='simple',
    )

    data_with_ner_tags = find_ner_tags(pipe, dataset)
    applied_data_ner_tags = apply_ner_tags(data_with_ner_tags)

    # data cleaning
    cleaned_dataset = clean_dataset(applied_data_ner_tags)

    with open(
        'dataset/cleaned-applied-ner-dataset.json',
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(
            cleaned_dataset,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()


# refer ke 3.2
