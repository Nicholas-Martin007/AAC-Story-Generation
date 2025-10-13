from peft import LoraConfig, TaskType


class FinetuneLora:
    def __init__(
        self,
        r,
        lora_alpha,
        lora_dropout,
    ):
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias='none',
            target_modules=[
                # 'q_proj',
                # 'k_proj',
                # 'v_proj',
                # 'o_proj',
                # 'gate_proj',
                # 'up_proj',
                # 'down_proj',
                'q',  # khusus t5
                'v',  # khusus t5
                'wi',  # khusus t5
                'wo',  # khusus t5
            ],
            modules_to_save=[
                'lm_head',
                'embed_tokens',
                'shared',  # khusus t5
            ],
        )

    def get_lora(self):
        return self.lora_config
