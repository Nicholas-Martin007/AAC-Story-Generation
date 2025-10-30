import os
import sys

sys.path.append(os.path.abspath('./'))

from prompts import *

from config import *


def build_prompt(type='Story'):
    samples = (
        LIST_STORIES_PROMPT[:5]
        if type == 'Story'
        else LIST_CARDS_PROMPT[:5]
    )
    samples = LIST_STORIES_PROMPT[:5]

    examples_text = ''
    for i, shot in enumerate(samples):
        content = shot['content']
        examples_text += f'{i + 1}.\n{type}: {content}\n\n'

    return (
        STORY_PROMPT + '\n' + examples_text
        if type == 'Story'
        else CARD_PROMPT + '\n' + examples_text
    )


def get_message(
    user_prompt: str,
    use_story_prompt: bool = False,
    use_card_prompt: bool = False,
):
    if use_story_prompt:
        system_prompt = build_prompt(type='Story')
    elif use_card_prompt:
        system_prompt = build_prompt(type='Card')
    else:
        system_prompt = None

    return [
        {
            'role': 'system',
            'content': system_prompt,
        },
        {
            'role': 'user',
            'content': user_prompt,
        },
    ]
