from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training


class FinetuneModel():
    def __init__(
        self, 
        tokenizer, 
        lora_config, 
        model_path, 
        device
    ):
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            quantization_config=self.quantization_config,
        )

        self.model.resize_token_embeddings(len(tokenizer))
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(
            self.model,
            lora_config,
        )

    def get_model(self):
        return self.model

