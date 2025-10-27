import os
import sys

sys.path.append(os.path.abspath('./'))
from cards_prompt import (
    CARD_PROMPT,
    LIST_CARDS_PROMPT,
)
from stories_prompt import (
    LIST_STORIES_PROMPT,
    STORY_PROMPT,
)

from config import *


def build_prompt_story(n):
    samples = LIST_STORIES_PROMPT[n][:N_SHOTS]

    examples_text = ''
    for i, shot in enumerate(samples):
        content = shot['content']
        examples_text += f'{i + 1}.\nStory: {content}\n\n'

    return STORY_PROMPT[n] + '\n' + examples_text


def build_prompt_card():
    samples = LIST_CARDS_PROMPT[:N_SHOTS]

    examples_text = ''
    for i, shot in enumerate(samples):
        content = shot['content']
        examples_text += f'{i + 1}.\nCards: {content}\n\n'

    return CARD_PROMPT + '\n' + examples_text


def get_message(
    user_prompt: str,
    n: str = None,
    use_story_prompt: bool = False,
    use_card_prompt: bool = False,
):
    if use_story_prompt:
        system_prompt = build_prompt_story(n)
    elif use_card_prompt:
        system_prompt = build_prompt_card()
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
