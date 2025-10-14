import json
import os
from datetime import datetime

import numpy as np
import torch
from evaluate import load
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
        model_type='seq2seq',  # 'seq2seq' for T5, 'causal' for LLaMA
        model_name='model',  # Add model name for unique filenames
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.output_dir = output_dir
        self.eval_counter = 0
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model_name = model_name.replace(
            '/', '_'
        )  # Sanitize model name for filenames

        # Choose appropriate TrainingArguments based on model type
        if model_type == 'seq2seq':
            self.training_args = Seq2SeqTrainingArguments(
                output_dir=output_dir,
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=2,
                num_train_epochs=n_epochs,
                weight_decay=weight_decay,
                optim='paged_adamw_32bit',
                logging_steps=100,
                warmup_ratio=0.01,
                eval_strategy='no',
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
                num_train_epochs=n_epochs,
                weight_decay=weight_decay,
                optim='paged_adamw_32bit',
                logging_steps=100,
                warmup_ratio=0.01,
                eval_strategy='no',
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
            )

    def compute_metrics(self, eval_preds):
        """For seq2seq models with predict_with_generate=True"""
        rouge = load('rouge')
        bertscore = load('bertscore')

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

        # Compute ROUGE
        rouge_result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        # Compute BERTScore with multilingual base model
        bertscore_result = bertscore.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            model_type='bert-base-multilingual-cased',
            lang='id',
        )

        # Calculate average BERTScore
        bertscore_avg = {
            'precision': float(
                np.mean(bertscore_result['precision'])
            ),
            'recall': float(np.mean(bertscore_result['recall'])),
            'f1': float(np.mean(bertscore_result['f1'])),
        }

        results = {
            'metadata': {
                'timestamp': datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S'
                ),
                'num_samples': len(decoded_preds),
                'eval_step': self.eval_counter,
                'model_name': self.model_name,
                'bertscore_model': 'bert-base-multilingual-cased',
            },
            'overall_metrics': {
                'ROUGE': {
                    'rouge1': float(rouge_result['rouge1']),
                    'rouge2': float(rouge_result['rouge2']),
                    'rougeL': float(rouge_result['rougeL']),
                },
                'BERTScore': bertscore_avg,
            },
            'samples': [],
        }

        # Save per-sample scores
        for i in range(min(10, len(decoded_preds))):
            sample = {
                'sample_id': i + 1,
                'reference': decoded_labels[i],
                'prediction': decoded_preds[i],
                'bertscore_precision': float(
                    bertscore_result['precision'][i]
                ),
                'bertscore_recall': float(
                    bertscore_result['recall'][i]
                ),
                'bertscore_f1': float(bertscore_result['f1'][i]),
            }
            results['samples'].append(sample)

        os.makedirs(self.output_dir, exist_ok=True)
        json_filename = os.path.join(
            self.output_dir,
            f'{self.model_name}_evaluation_step_{self.eval_counter}.json',
        )

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f'\nResults saved to: {json_filename}')
        print(f'ROUGE-L: {rouge_result["rougeL"]:.4f}')
        print(f'BERTScore F1: {bertscore_avg["f1"]:.4f}')
        print(f'{"=" * 80}\n')

        self.eval_counter += 1

        return {
            'rouge1': rouge_result['rouge1'],
            'rouge2': rouge_result['rouge2'],
            'rougeL': rouge_result['rougeL'],
            'bertscore_precision': bertscore_avg['precision'],
            'bertscore_recall': bertscore_avg['recall'],
            'bertscore_f1': bertscore_avg['f1'],
        }

    def evaluate_causal(self, test_data):
        """Manual evaluation for causal models (LLaMA, Mistral)"""
        rouge = load('rouge')
        bertscore = load('bertscore')

        self.model.eval()
        decoded_preds = []
        decoded_labels = []

        print(f'\n{"=" * 80}')
        print(
            f'Generating predictions for {len(test_data)} samples...'
        )

        with torch.no_grad():
            for i, example in enumerate(test_data):
                # Get the full text to parse prompt and response
                full_text = self.tokenizer.decode(
                    example['input_ids'],
                    skip_special_tokens=True,
                )

                # Split at "assistant" to separate prompt from expected response
                # Assuming format: "system...user...assistant\n\nExpected Response"
                if 'assistant' in full_text:
                    parts = full_text.split('assistant', 1)
                    prompt_text = (
                        parts[0] + 'assistant'
                    )  # Include "assistant" marker
                    expected_response = (
                        parts[1].strip()
                        if len(parts) > 1
                        else ''
                    )
                else:
                    # Fallback: use the full text as prompt
                    prompt_text = full_text
                    expected_response = ''

                # Tokenize only the prompt (without expected response)
                prompt_ids = self.tokenizer(
                    prompt_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                )['input_ids'].to(self.model.device)

                # Generate
                generated_ids = self.model.generate(
                    prompt_ids,
                    max_new_tokens=512,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,  # Greedy decoding for evaluation
                    temperature=1.0,
                )

                # Decode prediction (remove input prompt)
                generated_text = self.tokenizer.decode(
                    generated_ids[0][len(prompt_ids[0]) :],
                    skip_special_tokens=True,
                )
                decoded_preds.append(generated_text.strip())

                # Use the expected response as label
                decoded_labels.append(expected_response)

                if i == 0:
                    print(f'Prompt: {prompt_text[:200]}...')
                    print(
                        f'Sample prediction: {generated_text[:100]}...'
                    )
                    print(
                        f'Sample reference: {expected_response[:100]}...'
                    )

                if i % 100 == 0:
                    print(
                        f'Processed {i}/{len(test_data)} samples...'
                    )

        # Filter out empty predictions/labels
        valid_pairs = [
            (pred, label)
            for pred, label in zip(decoded_preds, decoded_labels)
            if pred.strip() and label.strip()
        ]

        if not valid_pairs:
            print(
                '⚠️ WARNING: No valid prediction-label pairs found!'
            )
            return {
                'eval_rouge1': 0.0,
                'eval_rouge2': 0.0,
                'eval_rougeL': 0.0,
                'eval_bertscore_precision': 0.0,
                'eval_bertscore_recall': 0.0,
                'eval_bertscore_f1': 0.0,
            }

        decoded_preds, decoded_labels = zip(*valid_pairs)
        decoded_preds = list(decoded_preds)
        decoded_labels = list(decoded_labels)

        print(
            f'\n✅ Valid pairs: {len(valid_pairs)}/{len(test_data)}'
        )

        # Compute ROUGE
        rouge_result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        bertscore_result = bertscore.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            model_type='bert-base-multilingual-cased',
            lang='id',
        )

        # Calculate average BERTScore
        bertscore_avg = {
            'precision': float(
                np.mean(bertscore_result['precision'])
            ),
            'recall': float(np.mean(bertscore_result['recall'])),
            'f1': float(np.mean(bertscore_result['f1'])),
        }

        # Save results
        results = {
            'metadata': {
                'timestamp': datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S'
                ),
                'num_samples': len(decoded_preds),
                'eval_step': self.eval_counter,
                'model_name': self.model_name,
                'bertscore_model': 'bert-base-multilingual-cased',
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
            },
            'overall_metrics': {
                'ROUGE': {
                    'rouge1': float(rouge_result['rouge1']),
                    'rouge2': float(rouge_result['rouge2']),
                    'rougeL': float(rouge_result['rougeL']),
                },
                'BERTScore': bertscore_avg,
            },
            'samples': [],
        }

        for i in range(min(10, len(decoded_preds))):
            sample = {
                'sample_id': i + 1,
                'reference': decoded_labels[i],
                'prediction': decoded_preds[i],
                'bertscore_precision': float(
                    bertscore_result['precision'][i]
                ),
                'bertscore_recall': float(
                    bertscore_result['recall'][i]
                ),
                'bertscore_f1': float(bertscore_result['f1'][i]),
            }
            results['samples'].append(sample)

        os.makedirs(self.output_dir, exist_ok=True)
        json_filename = os.path.join(
            self.output_dir,
            f'{self.model_name}_lr{self.learning_rate}_wd{self.weight_decay}_evaluation_step_{self.eval_counter}.json',
        )

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f'\nResults saved to: {json_filename}')
        print(f'ROUGE-L: {rouge_result["rougeL"]:.4f}')
        print(f'BERTScore F1: {bertscore_avg["f1"]:.4f}')
        print(f'{"=" * 80}\n')

        self.eval_counter += 1

        return {
            'eval_rouge1': rouge_result['rouge1'],
            'eval_rouge2': rouge_result['rouge2'],
            'eval_rougeL': rouge_result['rougeL'],
            'eval_bertscore_precision': bertscore_avg[
                'precision'
            ],
            'eval_bertscore_recall': bertscore_avg['recall'],
            'eval_bertscore_f1': bertscore_avg['f1'],
        }

    def get_trainer(self):
        return self.trainer
