import os
import sys

sys.path.append(os.path.abspath('./'))
import re

from setup import (
    generation_config,
    model,
    tokenizer,
)

from config import *


def format_output(
    response: str,
    use_story_format: bool = False,
    use_card_format: bool = False,
):
    if not isinstance(response, str):
        return None

    key = None
    if use_story_format:
        key = 'story'
    elif use_card_format:
        key = 'cards'

    if key:
        pattern = rf'(?:{key}):\s*(.*)'
        match = re.search(
            pattern,
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        else:
            return '[ERROR-MESSAGE]' + response


def generate_text(
    input_ids: torch.Tensor,
    use_story_format: bool = False,
    use_card_format: bool = False,
) -> str:
    "Output only without formatting"

    outputs = model.generate(
        input_ids,
        generation_config=generation_config,
    )

    response = outputs[0][input_ids.shape[-1] :]
    response = tokenizer.decode(
        response,
        skip_special_tokens=True,
    )

    output = format_output(
        response,
        use_story_format,
        use_card_format,
    )

    return output
