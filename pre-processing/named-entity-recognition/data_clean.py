import os
import sys

sys.path.append(os.path.abspath('./'))
import re

from config import *


def clean_dataset(
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
