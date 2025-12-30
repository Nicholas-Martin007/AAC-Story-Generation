# CLEAN DATA (STEP 3)

import json
import re
from typing import Iterator, List


import os
import sys

sys.path.append(os.path.abspath('./'))
from utils.file_utils import *


def save_file(
    data: List[str],
    save_filename: str,
    ensure_ascii: bool = True,
) -> None:
    with open(
        save_filename,
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=ensure_ascii,
        )


def remove_duplicate(
    data: Iterator[List[str]],
) -> Iterator[List[str]]:
    return list(
        set(data for k, v in data.items() for data in v['text'])
    )


def remove_text(data: Iterator[List[str]]) -> List[str]:
    result = []

    for text in data:
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text.split()) >= 3 and len(text) >= 20:
            result.append(text)

    return result


#############################
data = read_file(
    filename='download_cerpen/raw-text-cerpen.json',
)

data = remove_text(
    remove_duplicate(data=data),
)

save_file(
    data=data,
    save_filename='text-cerpen.json',
    ensure_ascii=False,
)
