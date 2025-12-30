import os
import sys

sys.path.append(os.path.abspath('./'))
import json
from typing import List

import pandas as pd

from config import *
from utils.file_utils import *


crime_keywords = [
    'bunuh',
    'mati',
    'tembak',
    'curi',
    'rampok',
    'perkosa',
    'mutilasi',
    'siksa',
    'hina',
    'ras',
    'kafir',
    'bodoh',
    'goblok',
    'overdosis',
    'depresi',
    'telanjang',
    'porno',
    'seks',
    'mesum',
    'persetubuhan',
    'sakit',
    'nangis',
    'kepedihan',
    'sedih',
]


def remove_duplicate(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return df.drop_duplicates(subset=['sentences'], keep='first')


def filter_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df[df['words'] >= 10]
    df = df[
        ~df['sentences'].str.contains(
            '|'.join(crime_keywords), case=False
        )
    ]

    return df


if __name__ == '__main__':
    data = read_file(filename='dataset/text-cerpen.json')

    df = pd.DataFrame(data, columns=['sentences'])

    # sentence -> word # untuk dapat length
    df['words'] = (
        df['sentences']
        .apply(lambda s: s.split())
        .apply(lambda w: len(w))
    )

    df = filter_data(
        remove_duplicate(df),
    )

    save_file(
        data=df['sentences'].tolist(),
        save_filename='dataset/pre-processed-dataset.json',
        ensure_ascii=False,
    )
