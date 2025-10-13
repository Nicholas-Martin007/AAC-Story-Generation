import json
import os
from datetime import datetime
from functools import partial

import numpy as np
import torch
from evaluate import load
from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


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
        self.model = model
        self.output_dir = output_dir
        self.eval_counter = 0

        self.training_args = (
            Seq2SeqTrainingArguments(  # Changed!
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
                eval_steps=500,
                eval_strategy='epoch',
                save_strategy='epoch',
                save_steps=500,
                fp16=True,
                lr_scheduler_type='cosine',
                gradient_checkpointing=True,
                dataloader_drop_last=True,
                report_to=None,
                predict_with_generate=True,  # Important for T5!
                generation_max_length=512,
            )
        )

        from transformers import DataCollatorForSeq2Seq

        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            max_length=512,
        )

        self.trainer = Seq2SeqTrainer(  # Changed!
            model=model,
            tokenizer=tokenizer,
            args=self.training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,  # No partial needed
        )

    def compute_metrics(self, eval_preds):
        rouge = load('rouge')
        bertscore = load('bertscore')

        predictions, labels = eval_preds

        # When predict_with_generate=True, predictions are already decoded token ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Replace -100 in labels
        labels = np.where(
            labels != -100, labels, self.tokenizer.pad_token_id
        )

        # Decode
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

        # # Compute BERTScore
        # bertscore_result = bertscore.compute(
        #     predictions=decoded_preds,
        #     references=decoded_labels,
        #     lang='id',
        #     model_type='bert-base-multilingual-cased',
        # )

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
                # 'BERTSCORE': {
                #     'precision': float(
                #         np.mean(bertscore_result['precision'])
                #     ),
                #     'recall': float(
                #         np.mean(bertscore_result['recall'])
                #     ),
                #     'f1': float(np.mean(bertscore_result['f1'])),
                # },
            },
            'samples': [],
        }

        for i in range(min(10, len(decoded_preds))):
            sample = {
                'sample_id': i + 1,
                'reference': decoded_labels[i],
                'prediction': decoded_preds[i],
                # 'bertscore_f1': float(bertscore_result['f1'][i]),
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
        # print(
        #     f'BERTScore F1: {np.mean(bertscore_result["f1"]):.4f}'
        # )
        print(f'{"=" * 80}\n')

        self.eval_counter += 1

        return {
            'rouge1': rouge_result['rouge1'],
            'rouge2': rouge_result['rouge2'],
            'rougeL': rouge_result['rougeL'],
            # 'bertscore_f1': float(
            #     np.mean(bertscore_result['f1'])
            # ),
        }

    def get_trainer(self):
        return self.trainer
