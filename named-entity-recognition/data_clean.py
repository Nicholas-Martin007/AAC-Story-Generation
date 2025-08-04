import sys
import os

sys.path.append(os.path.abspath('./'))
from config import *

import re
import json


def clean_dataset(
    dataset: list[str],
) -> list[str]:
    """
    Remove typos on <|PER|>,
    Example: "aat<|PER|>", "<|PER|>1"
    """
    result = []
    for data in dataset:
        matches = re.finditer(
            r'<\|PER\|>([a-zA-Z0-9]+)\b',
            data,
        )

        sorted_matches = sorted(
            matches,
            key=lambda x: x.start(1),
            reverse=True,
        )

        for match in sorted_matches:
            start, end = (
                match.start(1),
                match.end(1),
            )

            data = data[:start] + data[end:]

        result.append(data)

    return result
