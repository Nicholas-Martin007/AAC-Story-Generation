from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
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
        self.eval_counter = 0
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model_name = model_name.replace('/', '_')

        # Choose appropriate TrainingArguments based on model type
        if model_type == 'seq2seq':
            self.training_args = Seq2SeqTrainingArguments(
                output_dir=output_dir,
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                # gradient_accumulation_steps=2,
                num_train_epochs=n_epochs,
                weight_decay=weight_decay,
                optim='paged_adamw_32bit',
                logging_steps=100,
                warmup_ratio=0.01,
                eval_strategy='epoch',  # Changed from 'no' to 'epoch'
                save_strategy='epoch',
                fp16=True,
                lr_scheduler_type='cosine',
                gradient_checkpointing=True,
                dataloader_drop_last=True,
                report_to=None,
                # Removed predict_with_generate since we're using perplexity
            )
        else:  # causal
            self.training_args = TrainingArguments(
                output_dir=output_dir,
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                # gradient_accumulation_steps=2,
                num_train_epochs=n_epochs,
                weight_decay=weight_decay,
                optim='paged_adamw_32bit',
                logging_steps=100,
                warmup_ratio=0.01,
                eval_strategy='epoch',  # Changed from 'no' to 'epoch'
                save_strategy='epoch',
                fp16=True,
                lr_scheduler_type='cosine',
                gradient_checkpointing=True,
                dataloader_drop_last=True,
                report_to=None,
            )

        # Data collator
        if model_type == 'seq2seq':
            from transformers import DataCollatorForSeq2Seq

            self.data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding=True,
                max_length=512,
            )
        else:  # causal
            from transformers import (
                DataCollatorForLanguageModeling,
            )

            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8,
            )

        # Choose appropriate Trainer
        if model_type == 'seq2seq':
            self.trainer = Seq2SeqTrainer(
                model=model,
                tokenizer=tokenizer,
                args=self.training_args,
                train_dataset=train_data,
                eval_dataset=test_data,
                data_collator=self.data_collator,
                # compute_metrics=self.compute_metrics_perplexity,  # Use perplexity
            )
        else:  # causal
            self.trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=self.training_args,
                train_dataset=train_data,
                eval_dataset=test_data,
                data_collator=self.data_collator,
                # compute_metrics=self.compute_metrics_perplexity,  # Use perplexity
            )

    def compute_metrics_perplexity(self, eval_preds):
        """Compute perplexity from loss - much faster than generation-based metrics"""
        predictions, labels = eval_preds

        # For seq2seq, predictions is (loss, logits, ...)
        # For causal, predictions is just logits
        # But we don't actually need predictions here since loss is computed automatically

        # The loss is already computed by the Trainer, we just format the output
        results = {
            'eval_perplexity_note': 'Perplexity = exp(loss), see eval_loss in logs'
        }

        print(f'\n{"=" * 80}')
        print(f'Evaluation step {self.eval_counter} completed')
        print(
            'Perplexity will be calculated from eval_loss: perplexity = exp(eval_loss)'
        )
        print(f'{"=" * 80}\n')

        self.eval_counter += 1

        return results

    # COMMENTED OUT: Original ROUGE/BERTScore evaluation
    # def compute_metrics(self, eval_preds):
    #     """For seq2seq models with predict_with_generate=True"""
    #     rouge = load('rouge')
    #     bertscore = load('bertscore')
    #
    #     predictions, labels = eval_preds
    #     if isinstance(predictions, tuple):
    #         predictions = predictions[0]
    #
    #     labels = np.where(
    #         labels != -100, labels, self.tokenizer.pad_token_id
    #     )
    #
    #     decoded_preds = self.tokenizer.batch_decode(
    #         predictions, skip_special_tokens=True
    #     )
    #     decoded_labels = self.tokenizer.batch_decode(
    #         labels, skip_special_tokens=True
    #     )
    #
    #     # ... rest of ROUGE/BERTScore computation

    # COMMENTED OUT: Causal model evaluation with generation
    # def evaluate_causal(self, test_data):
    #     """Manual evaluation for causal models (LLaMA, Mistral)"""
    #     rouge = load('rouge')
    #     bertscore = load('bertscore')
    #
    #     # ... generation and metric computation

    def get_trainer(self):
        return self.trainer
