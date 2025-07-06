import json
import re

dataset_path = "data_generation\web_scrapping\output.json"

def clean_data(text):
    text = re.sub(r"\s+", " ", text).strip()
    if(len(text.split()) < 2):
        return None

    return text

def get_data():
    with open(f"{dataset_path}", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    texts = list(set(data for k, v in dataset.items() for data in v["text"]))
    texts = [clean_data(text) for text in texts]
    texts = [x for x in texts if x]
    

    return texts