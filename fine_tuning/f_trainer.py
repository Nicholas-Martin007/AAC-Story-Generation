import json
import os
from datetime import datetime
from functools import partial

import numpy as np
import torch
from evaluate import load
from transformers import (
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
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
        model_type='seq2seq',  # 'seq2seq' for T5, 'causal' for LLaMA
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.output_dir = output_dir
        self.eval_counter = 0
        self.model_type = model_type

        # Choose appropriate TrainingArguments based on model type
        if model_type == 'seq2seq':
            self.training_args = Seq2SeqTrainingArguments(
                output_dir=output_dir,
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=2,
                num_train_epochs=1,
                weight_decay=weight_decay,
                optim='paged_adamw_32bit',
                logging_steps=100,
                warmup_ratio=0.01,
                eval_strategy='epoch',
                save_strategy='epoch',
                fp16=True,
                lr_scheduler_type='cosine',
                gradient_checkpointing=True,
                dataloader_drop_last=True,
                report_to=None,
                predict_with_generate=True,  # For T5
                generation_max_length=512,
            )
        else:  # causal
            self.training_args = TrainingArguments(
                output_dir=output_dir,
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=2,
                num_train_epochs=1,
                weight_decay=weight_decay,
                optim='paged_adamw_32bit',
                logging_steps=100,
                warmup_ratio=0.01,
                eval_strategy='epoch',
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
                compute_metrics=self.compute_metrics,
            )
        else:  # causal
            self.trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=self.training_args,
                train_dataset=train_data,
                eval_dataset=test_data,
                data_collator=self.data_collator,
                # For causal models, we'll do manual generation in evaluate
            )

    def compute_metrics(self, eval_preds):
        """For seq2seq models with predict_with_generate=True"""
        rouge = load('rouge')

        predictions, labels = eval_preds

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        labels = np.where(
            labels != -100, labels, self.tokenizer.pad_token_id
        )

        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [
            label.strip() for label in decoded_labels
        ]

        print(f'\n{"=" * 80}')
        print(
            f'Computing metrics for {len(decoded_preds)} samples...'
        )
        print(f'Sample prediction: {decoded_preds[0][:100]}...')
        print(f'Sample reference: {decoded_labels[0][:100]}...')

        rouge_result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        results = {
            'metadata': {
                'timestamp': datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S'
                ),
                'num_samples': len(decoded_preds),
                'eval_step': self.eval_counter,
            },
            'overall_metrics': {
                'ROUGE': {
                    'rouge1': float(rouge_result['rouge1']),
                    'rouge2': float(rouge_result['rouge2']),
                    'rougeL': float(rouge_result['rougeL']),
                },
            },
            'samples': [],
        }

        for i in range(min(10, len(decoded_preds))):
            sample = {
                'sample_id': i + 1,
                'reference': decoded_labels[i],
                'prediction': decoded_preds[i],
            }
            results['samples'].append(sample)

        os.makedirs(self.output_dir, exist_ok=True)
        json_filename = os.path.join(
            self.output_dir,
            f'evaluation_results_step_{self.eval_counter}.json',
        )
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f'\nResults saved to: {json_filename}')
        print(f'ROUGE-L: {rouge_result["rougeL"]:.4f}')
        print(f'{"=" * 80}\n')

        self.eval_counter += 1

        return {
            'rouge1': rouge_result['rouge1'],
            'rouge2': rouge_result['rouge2'],
            'rougeL': rouge_result['rougeL'],
        }

    def evaluate_causal(self, test_data):
        """Manual evaluation for causal models (LLaMA, Mistral)"""
        rouge = load('rouge')

        self.model.eval()
        decoded_preds = []
        decoded_labels = []

        print(f'\n{"=" * 80}')
        print(
            f'Generating predictions for {len(test_data)} samples...'
        )

        with torch.no_grad():
            for i, example in enumerate(test_data):
                # Get input and label
                input_ids = (
                    torch.tensor(example['input_ids'])
                    .unsqueeze(0)
                    .to(self.model.device)
                )
                labels = example['labels']

                # Generate
                generated_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=512,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # Decode prediction (remove input prompt)
                generated_text = self.tokenizer.decode(
                    generated_ids[0][len(input_ids[0]) :],
                    skip_special_tokens=True,
                )
                decoded_preds.append(generated_text.strip())

                # Decode label
                label_ids = [l for l in labels if l != -100]
                label_text = self.tokenizer.decode(
                    label_ids, skip_special_tokens=True
                )
                decoded_labels.append(label_text.strip())

                if i == 0:
                    print(
                        f'Sample prediction: {generated_text[:100]}...'
                    )
                    print(
                        f'Sample reference: {label_text[:100]}...'
                    )

        # Compute ROUGE
        rouge_result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        # Save results
        results = {
            'metadata': {
                'timestamp': datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S'
                ),
                'num_samples': len(decoded_preds),
                'eval_step': self.eval_counter,
            },
            'overall_metrics': {
                'ROUGE': {
                    'rouge1': float(rouge_result['rouge1']),
                    'rouge2': float(rouge_result['rouge2']),
                    'rougeL': float(rouge_result['rougeL']),
                },
            },
            'samples': [],
        }

        for i in range(min(10, len(decoded_preds))):
            sample = {
                'sample_id': i + 1,
                'reference': decoded_labels[i],
                'prediction': decoded_preds[i],
            }
            results['samples'].append(sample)

        os.makedirs(self.output_dir, exist_ok=True)
        json_filename = os.path.join(
            self.output_dir,
            f'evaluation_results_step_{self.eval_counter}.json',
        )
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f'\nResults saved to: {json_filename}')
        print(f'ROUGE-L: {rouge_result["rougeL"]:.4f}')
        print(f'{"=" * 80}\n')

        self.eval_counter += 1

        return {
            'eval_rouge1': rouge_result['rouge1'],
            'eval_rouge2': rouge_result['rouge2'],
            'eval_rougeL': rouge_result['rougeL'],
        }

    def get_trainer(self):
        return self.trainer
