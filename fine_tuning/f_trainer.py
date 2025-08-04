import numpy as np
from evaluate import load
from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer


class FinetuneTrainer:
    def __init__(
        self,
        train_data,
        test_data,
        model,
        tokenizer,
        lora_config,
        output_dir,
        learning_rate,
        weight_decay,
        batch_size,
    ):
        self.tokenizer = tokenizer
        self.rouge = load('rouge')
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,
            num_train_epochs=5,
            weight_decay=weight_decay,  # regularization
            optim='paged_adamw_32bit',
            logging_steps=100,
            warmup_ratio=0.03,  # 5%
            eval_steps=500,
            eval_strategy='steps',
            save_strategy='steps',
            save_steps=500,
            fp16=True,
            lr_scheduler_type='cosine',
            gradient_checkpointing=True,
            dataloader_drop_last=True,
            report_to=None,
        )

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )

        self.trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=self.training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            max_seq_length=512,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            packing=False,
            peft_config=lora_config,
        )

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(
            preds != -100,
            preds,
            self.tokenizer.pad_token_id,
        )
        labels = np.where(
            labels != -100,
            labels,
            self.tokenizer.pad_token_id,
        )

        decoded_preds = [
            p.strip()
            for p in self.tokenizer.batch_decode(
                preds,
                skip_special_tokens=True,
            )
        ]
        decoded_labels = [
            l.strip()
            for l in self.tokenizer.batch_decode(
                labels,
                skip_special_tokens=True,
            )
        ]

        result = self.rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        # result = {k: v for k, v in result.items()}
        print('RESULT:', result)
        return result

    def preprocess_logits_for_metrics(self, logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def get_trainer(self):
        return self.trainer
