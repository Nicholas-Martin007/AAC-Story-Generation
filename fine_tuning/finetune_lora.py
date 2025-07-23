from peft import LoraConfig, TaskType

class FinetuneLora:
    def __init__(self):
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=64,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            modules_to_save=["lm_head", "embed_tokens"]
        )

    def get_lora(self):
        return self.lora_config