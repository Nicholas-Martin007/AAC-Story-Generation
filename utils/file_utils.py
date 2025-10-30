import json
from typing import List


def read_file(
    filename: str,
) -> List[str]:
    with open(
        filename,
        'r',
    ) as f:
        data = json.load(f)

    return data


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
