from get_data import get_data
from message_formatter import get_message
from setup import tokenizer, model
from model import generate_text

if __name__ == "__main__":
    dataset = get_data()
    generated_data = []

    for data in dataset:
        messages = get_message(example=data)

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        output = generate_text(input_ids=input_ids)
        generated_data.append(output)




    # with open("train_data.json", "w", encoding="utf-8") as f:
    #     f.write(generated_data)
