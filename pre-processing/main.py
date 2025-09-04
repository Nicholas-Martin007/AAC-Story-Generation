import os
import sys

sys.path.append(os.path.abspath('./'))
import json

import pandas as pd

from config import *

with open(f'{DATA_PATH}', encoding='utf8') as f:
    data = json.load(f)

df = pd.DataFrame(data, columns=['sentences'])

# sentence -> word
df['words'] = (
    df['sentences']
    .apply(lambda s: s.split())
    .apply(lambda w: len(w))
)

# df['words'].describe()


# (NOT WORK)
# q = [df['words'].quantile(1 / 4), df['words'].quantile(3 / 4)]

# # based on paper, using iqr outlier
# iqr = q[1] - q[0]
# iqr_low = q[0] - (3 / 2) * iqr
# iqr_high = q[1] + (3 / 2) * iqr

# df = df[(df['words'] < iqr_low) | (df['words'] > iqr_high)]


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
min_words_length = 10


# remove duplicate
df = df.drop_duplicates(subset=['sentences'], keep='first')

# filtering
df = df[df['words'] >= min_words_length]
df = df[
    ~df['sentences'].str.contains(
        '|'.join(crime_keywords), case=False
    )
]

with open(
    'dataset/pre-processed-dataset.json', 'w', encoding='utf-8'
) as f:
    json.dump(
        df['sentences'].tolist(),
        f,
        ensure_ascii=False,
        indent=2,
    )
print()
