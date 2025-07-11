import json
import re
from config import DATA_PATH

def clean_data(dataset):
    '''
    Input: Dict
    Ouput: List[String]
    '''
    texts = list(set(data for k, v in dataset.items() for data in v["text"]))

    cleaned_texts = []

    for text in texts:
        text = re.sub(r"\s+", " ", text).strip()
        if(len(text.split()) >= 2):
            cleaned_texts.append(text)

    return cleaned_texts

def save_texts(texts):
    with open("texts.json", "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

def get_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    cleaned_texts = clean_data(dataset)
    save_texts(cleaned_texts)
