import torch

DATA_PATH = "text-cerpen.json"
APPLIED_NER_DATA_PATH = "applied-ner-dataset.json"
NER_MODEL_PATH = "cahya/NusaBert-ner-v1.3"
N_SHOTS = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
