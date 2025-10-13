from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

models_causal = [
    'meta-llama/Llama-3.2-1B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
]

models_seq2seq = [
    'google/flan-t5-large',
    'google/flan-t5-xl',
]

for model_id in models_causal:
    print(f'Downloading causal: {model_id}')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    AutoModelForCausalLM.from_pretrained(model_id)

for model_id in models_seq2seq:
    print(f'Downloading seq2seq: {model_id}')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    AutoModelForSeq2SeqLM.from_pretrained(model_id)
