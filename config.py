import torch

DATA_PATH = "dataset/text-cerpen.json"
APPLIED_NER_DATA_PATH = "dataset/applied-ner-dataset.json"
CLEANED_APPLIED_NER_DATA_PATH = "dataset/cleaned-applied-ner-dataset.json"
AAC_CARD_PATH = "dataset/aac_card_dataset.json"
AAC_STORY_PATH = "dataset/aac_story_dataset.json"

NER_MODEL_PATH = "cahya/NusaBert-ner-v1.3"
N_SHOTS = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SDG_MODEL_PATH = "/home/dev/Downloads/Meta-Llama-3.1-8B-Instruct/"

MODEL_PATH = "/home/dev/Downloads/Llama-3.2-3B-Instruct/"

FINE_TUNE_OUTPUT_DIR = "./training_output"

SEED = 5589
# "google/flan-t5-small"