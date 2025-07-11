
from config import DEVICE, MODEL_CONFIG
from get_data import get_data
from message_formatter import get_message
from setup import tokenizer, model
from model import generate_text
import json
import time

def read_file(filepath="documentation_files/experiment-2/story-generation/story-temp0.9.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    return dataset

if __name__ == "__main__":
    # get_data() # RUN FIRST TIME
    dataset = read_file()

    tes_100 = dataset[:100]
    generated_data = []
    for data in tes_100:
        start = time.time()

        messages = get_message(data)

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        output = generate_text(input_ids=input_ids)
        
        end = time.time()
        print(f"Generated in {end - start:.2f} seconds")
        generated_data.append(output)
        
        
        print(f"AAC Cards: {output}")
        print("\n\n")


    temp=f"{MODEL_CONFIG['generation']['temperature']}"
    top_p=f"{MODEL_CONFIG['generation']['top_p']}"

    with open(f"train_data-temp{temp}-top_p{top_p}.json", "w", encoding="utf-8") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=2)
