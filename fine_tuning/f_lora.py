from peft import LoraConfig, TaskType


class FinetuneLora:
    def __init__(
        self,
        r,
        lora_alpha,
        lora_dropout,
        model_type,
    ):
        if model_type == 'seq2seq':
            task_type = TaskType.SEQ_2_SEQ_LM
            target_modules = [
                'q',
                'v',
                'wi',
                'wo',
                'shared',
            ]
        else:
            task_type = TaskType.CAUSAL_LM
            target_modules = [
                'q_proj',
                'k_proj',
                'v_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj',
            ]

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
