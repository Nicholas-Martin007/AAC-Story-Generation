import numpy as np
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from evaluate import load as load_metric

class FinetuneTrainer:
    def __init__(
        self,
        output_dir,
        model,
        tokenizer,
        train_data,
        test_data,
        lora_config,
    ):  
        self.tokenizer = tokenizer
        self.rouge_score = load_metric("rouge")
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            weight_decay=0.01,
            optim="paged_adamw_32bit",
            logging_steps=100,
            eval_steps=100,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=500,
            fp16=True,
            lr_scheduler_type="cosine",
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
            peft_config=lora_config
        )

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_preds = [p.strip() for p in self.tokenizer.batch_decode(preds, skip_special_tokens=True)]
        decoded_labels = [l.strip() for l in self.tokenizer.batch_decode(labels, skip_special_tokens=True)]

        result = self.rouge_score.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        print("RESULT:", result)
        return result

    def preprocess_logits_for_metrics(self, logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)
    
    def get_trainer(self):
        return self.trainer