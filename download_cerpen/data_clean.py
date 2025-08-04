import json
import re


dataset_path = 'download_cerpen/raw-text-cerpen.json'

with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

removed_duplicate_dataset = list(
    set(data for k, v in dataset.items() for data in v['text'])
)

cleaned_dataset = []

for text in removed_duplicate_dataset:
    text = re.sub(r'\s+', ' ', text).strip()

    if len(text.split()) >= 3 and len(text) >= 20:
        cleaned_dataset.append(text)

with open(
    'text-cerpen.json',
    'w',
    encoding='utf-8',
) as f:
    json.dump(
        cleaned_dataset,
        f,
        ensure_ascii=False,
        indent=2,
    )

print()
