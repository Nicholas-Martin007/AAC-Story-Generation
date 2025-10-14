from transformers import AutoTokenizer


class FinetuneTokenizer:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )

        # Untuk FLAN-T5, pastikan tokenizer memiliki pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_tokenizer(self):
        return self.tokenizer
