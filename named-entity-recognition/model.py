import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


def manual_ner(
    tokenizer: AutoTokenizer, model: AutoModelForTokenClassification, sentence: str
) -> None:
    """
    To see details of the NER model
    """
    colors_list = [
        "255;99;132",
        "54;162;235",
        "255;206;86",
        "75;192;192",
        "153;102;255",
        "255;159;64",
        "199;199;199",
        "83;102;255",
        "255;102;255",
        "102;255;178",
        "255;204;153",
        "204;255;102",
        "255;153;204",
        "102;153;255",
        "255;255;102",
        "102;255;255",
        "204;153;255",
        "255;102;153",
        "153;255;204",
        "178;102;255",
        "255;102;102",
        "102;255;153",
        "255;178;102",
        "153;255;255",
        "255;153;102",
        "204;255;255",
        "102;204;255",
        "255;255;204",
        "204;204;255",
        "255;204;229",
        "204;255;229",
        "255;229;204",
        "229;204;255",
        "255;255;255",
        "192;192;192",
        "255;51;153",
        "153;255;51",
        "51;153;255",
        "0;0;0",
    ]

    tokenized_input = tokenizer(
        sentence,
        # is_split_into_words=True,
        return_tensors="pt",
    )
    input_ids = tokenized_input["input_ids"][0]
    with torch.no_grad():
        outputs = model(**tokenized_input)

    id2label = model.config.id2label

    texts = tokenizer.convert_ids_to_tokens(input_ids)
    labels = outputs.logits.argmax(axis=2)[0].tolist()

    for text, label in zip(texts, labels):
        print(
            text
            + f"\x1b[0;30;48;2;{colors_list[label % len(colors_list)]}m[{id2label[label]}]\x1b[0m",
            end=" ",
        )

    print("\n\n")


def concat_entity(outputs):
    """
    Concat all entities to prevent duplication
    """
    storage = []
    prev = None

    for entity in outputs:
        curr_start, curr_end = entity["start"], entity["end"]
        curr_entity = entity["entity_group"]

        if (
            prev is None
            or prev["entity_group"] != curr_entity
            or curr_start != prev["end"]
        ):
            if prev is not None:
                storage.append(prev)

            prev = {"start": curr_start, "end": curr_end, "entity_group": curr_entity}

        else:
            prev["end"] = curr_end

    if prev is not None:
        storage.append(prev)

    return storage


def insert_entity_tags(storage: list[str], sentence: str) -> str:
    """
    insert entities into sentences, (PER tag only)
    """
    sorted_storage = sorted(storage, key=lambda x: x["start"], reverse=True)

    for store_data in sorted_storage:
        start, end = store_data["start"], store_data["end"]
        entity = store_data["entity_group"]

        if entity == "PER":
            sentence = sentence[:start] + f" <|{entity}|>" + sentence[end:]

    return sentence


def apply_ner_tags(pipe: pipeline, dataset: list[str]) -> list[str]:
    """
    Apply NER tags into the sentences
    """
    result = []

    for data in dataset:
        outputs = pipe(data)

        if not outputs:
            result.append(data)
            continue

        storage = concat_entity(outputs)
        tagged_sentence = insert_entity_tags(storage, data)
        result.append(tagged_sentence)

    return result
