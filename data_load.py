import json
import pandas as pd
import numpy as np
from datasets import Dataset
import uuid

data_path = "nic-learn/skripsi_nic/synthetic_data_generation/train_data-temp0.9-top_p0.9.json"

with open(f"{data_path}", "r", encoding="utf-8") as f:
    dataset = json.load(f)

messages = [data["messages"] for data in dataset]


formatted_data = []
for i, pair in enumerate(messages):
    formatted_data.append({
        "prompt": pair[0]["content"],
        "prompt_id": str(uuid.uuid4()),
        "messages": pair
    })

hf_dataset = Dataset.from_list(formatted_data)
hf_dataset.save_to_disk("cerpen_dataset")
print()