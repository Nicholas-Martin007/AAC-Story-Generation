import sys
import os

sys.path.append(os.path.abspath('./'))
from config import *

import json

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)
from model import apply_ner_tags
from data_clean import clean_dataset

if __name__ == '__main__':
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
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

    applied_ner_dataset = apply_ner_tags(pipe, dataset)

    with open(
        'applied-ner-dataset.json',
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(
            applied_ner_dataset,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # =============================================
    # CLEAN NER TYPO
    with open(
        APPLIED_NER_DATA_PATH,
        'r',
        encoding='utf-8',
    ) as f:
        dataset = json.load(f)

    cleaned_dataset = clean_dataset(dataset)

    with open(
        'cleaned-applied-ner-dataset.json',
        'w',
        encoding='utf-8',
    ) as f:
        json.dumps(
            cleaned_dataset,
            ensure_ascii=False,
            indent=2,
        )

    print()
