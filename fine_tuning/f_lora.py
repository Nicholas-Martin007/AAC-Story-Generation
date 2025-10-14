from peft import LoraConfig, TaskType


class FinetuneLora:
    def __init__(
        self,
        r,
        lora_alpha,
        lora_dropout,
        model_type,
    ):
        # Konfigurasi LoRA khusus untuk FLAN-T5
        task_type = TaskType.SEQ_2_SEQ_LM

        # Target modules untuk T5
        target_modules = ['q', 'v', 'k', 'o', 'wi', 'wo']

        self.lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias='none',
            target_modules=target_modules,
        )

    def get_lora(self):
        return self.lora_config
