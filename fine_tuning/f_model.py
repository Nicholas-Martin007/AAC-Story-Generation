import torch
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)


class FinetuneModel:
    def __init__(
        self,
        tokenizer,
        model_path,
        device,
        lora_config=None,
        model_type='seq2seq',
    ):
        # Konfigurasi kuantisasi untuk FLAN-T5
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )

        # Load model FLAN-T5
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            quantization_config=self.quantization_config,
            device_map='auto',
            trust_remote_code=True,
        )

        # Resize token embeddings jika diperlukan
        self.model.resize_token_embeddings(len(tokenizer))

    def insert_lora(self, lora_config):
        # Prepare model untuk training
        self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

    def get_model(self):
        return self.model
