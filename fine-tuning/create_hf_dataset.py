import json
import pandas as pd


with open("documentation_files/experiment-2/aac_cards/cards-temp0.3---story-temp-0.9.json", "r", encoding="utf-8") as f:
    cards = json.load(f)

with open("documentation_files/experiment-2/story-generation/story-temp0.9.json", "r", encoding="utf-8") as f:
    story = json.load(f)

df = pd.DataFrame(
    {
        "cards": cards,
        "story": story,
    }
)

df["cards"] = df["cards"].apply(lambda x: x if isinstance(x, list) and len(x) <= 30 else None)
df = df.dropna()

messages = []

for cards, story in zip(df["cards"], df["story"]):
    messages.append(
        [
            {
                "content": ", ".join(cards),
                "role": "user"
            },
            {
                "content": story,
                "role": "assistant"
            }
        ]
    )


from datasets import Dataset
import uuid

formatted_data = []
for i, pair in enumerate(messages):
    formatted_data.append(
        {
            "prompt": pair[0]["content"],
            "prompt_id": str(uuid.uuid4()),
            "messages": pair 
        }
    )

hf_dataset = Dataset.from_list(formatted_data)


hf_dataset.save_to_disk('./aac_cards_dataset')

print()