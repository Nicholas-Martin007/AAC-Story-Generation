import torch

N_PERSON = [
    'no_person',
    # 'one_person',
    # 'two_person',
    # 'three_person',
    # 'four_person',
    # 'five_person',
]

DATA_PATH = 'dataset/text-cerpen.json'
PRE_PROCESSED_DATA_PATH = 'dataset/pre-processed-dataset.json'
APPLIED_NER_DATA_PATH = 'dataset/applied-ner-dataset.json'
CLEANED_APPLIED_NER_DATA_PATH = (
    'dataset/cleaned-applied-ner-dataset.json'
)
AAC_CARD_PATH = 'dataset/aac_card_dataset.json'
AAC_STORY_PATH = 'dataset/aac_story_dataset.json'

NER_MODEL_PATH = 'cahya/NusaBert-ner-v1.3'
N_SHOTS = 5
DEVICE = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)

SDG_MODEL_PATH = (
    '/home/dev/Downloads/Meta-Llama-3.1-8B-Instruct/'
)

MODEL_PATH = {
    'llama3.2-1b': 'meta-llama/Llama-3.2-1B-Instruct',  # 1.24B parameters
    'llama3.2-3b': '/home/dev/Downloads/Llama-3.2-3B-Instruct/',  # 3.21B parameters
    'mistral7b': '/home/dev/Downloads/Mistral-7B-Instruct-v0.3/',  # 7.25B parameters
    'flan-large': 'google/flan-t5-large',  # 783M parameters
    'flan-xl': 'google/flan-t5-xl',  # 2.85B parameters
}

FINE_TUNE_OUTPUT_DIR = './training_output'

SEED = 42
# "google/flan-t5-small"
