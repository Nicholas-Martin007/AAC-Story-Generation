from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)


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
        n_epochs,
        model_type='seq2seq',
        model_name='model',
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.output_dir = output_dir
        self.model_type = model_type

        # ---- Training Arguments ----
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=n_epochs,
            weight_decay=weight_decay,
            optim='adafactor',
            logging_steps=10,
            warmup_steps=100,
            eval_strategy='no',  # Matikan evaluation sementara untuk debugging
            save_strategy='epoch',
            fp16=False,  # Matikan FP16 dulu untuk debugging
            lr_scheduler_type='linear',
            gradient_checkpointing=False,
            dataloader_drop_last=False,
            report_to=None,
            remove_unused_columns=True,
        )

        # ---- Data Collator sederhana ----
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding='longest',  # Gunakan 'longest' untuk dynamic padding
            max_length=128,
            label_pad_token_id=-100,
            return_tensors='pt',
        )

        # ---- Trainer ----
        self.trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=self.training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            data_collator=self.data_collator,
        )

    def get_trainer(self):
        return self.trainer
