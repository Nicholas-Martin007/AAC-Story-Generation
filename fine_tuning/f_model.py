import torch
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)


class FinetuneModel:
    def __init__(
        self,
        tokenizer,
        model_path,
        device,
        model_type='seq2seq',
    ):
        # DQ + NF4
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )

        if model_type == 'seq2seq':
            self.model = AutoModelForSeq2SeqLM.from_pretrained(  # khusus t5
                model_path,
                device_map=device,
                quantization_config=self.quantization_config,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                quantization_config=self.quantization_config,
            )

        self.model.resize_token_embeddings(len(tokenizer))

    def insert_lora(self, lora_config):
        self.model = prepare_model_for_kbit_training(
            self.model,
        )  # preprocess, required for quantized model
        self.model = get_peft_model(
            self.model,
            lora_config,
        )  # insert qlora

    def get_model(self):
        return self.model
