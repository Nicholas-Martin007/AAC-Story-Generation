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
        self.eval_counter = 0  # Track evaluation calls

        self.training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,  # untuk gpu efficiency
            num_train_epochs=5,
            weight_decay=weight_decay,  # regularization
            optim='paged_adamw_32bit',
            logging_steps=100,
            warmup_ratio=0.01,  # 5%
            eval_steps=500,
            eval_strategy='steps',
            save_strategy='steps',
            save_steps=500,
            fp16=True,
            lr_scheduler_type='cosine',
            gradient_checkpointing=True,
            dataloader_drop_last=True,
            report_to=None,
            predict_with_generate=True,
            generation_max_length=512,
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
            compute_metrics=partial(
                self.compute_metrics, tokenizer=tokenizer
            ),
            max_seq_length=512,
            packing=False,
            peft_config=lora_config,
        )

    def compute_perplexity(self, decoded_preds):
        try:
            perplexities = []
            for pred in decoded_preds:
                if len(pred.strip()) == 0:
                    continue

                # Tokenize prediction
                inputs = self.tokenizer(
                    pred,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                ).to(self.model.device)

                # Get loss
                with torch.no_grad():
                    outputs = self.model(
                        **inputs, labels=inputs['input_ids']
                    )
                    loss = outputs.loss

                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)

            avg_perplexity = (
                np.mean(perplexities) if perplexities else None
            )
            return avg_perplexity, perplexities

        except Exception as e:
            print(f'Error computing perplexity: {e}')
            return None, []

    def compute_metrics(self, eval_preds, tokenizer):
        rouge = load('rouge')
        bertscore = load('bertscore')

        predictions, labels = eval_preds

        print('\n' + '=' * 80)
        print('Computing evaluation metrics...')
        print('=' * 80)

        decoded_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )

        labels = np.where(
            labels != -100, labels, tokenizer.pad_token_id
        )
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [
            label.strip() for label in decoded_labels
        ]

        print(f'Total samples: {len(decoded_preds)}')

        print('Computing ROUGE scores...')
        rouge_result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        print('Computing BERTScore...')
        bertscore_result = bertscore.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            lang='id',
            model_type='bert-base-multilingual-cased',
        )

        print('Computing Perplexity...')
        avg_perplexity, perplexities = self.compute_perplexity(
            decoded_preds
        )

        # Prepare data for JSON
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
                'BERTSCORE': {
                    'precision': float(
                        np.mean(bertscore_result['precision'])
                    ),
                    'recall': float(
                        np.mean(bertscore_result['recall'])
                    ),
                    'f1': float(np.mean(bertscore_result['f1'])),
                },
                'PERPLEXITY': {
                    'average': float(avg_perplexity)
                    if avg_perplexity is not None
                    else None
                },
            },
            'samples': [],
        }

        # Add individual samples
        for i in range(len(decoded_preds)):
            sample = {
                'sample_id': i + 1,
                'Reference': decoded_labels[i],
                'Prediction': decoded_preds[i],
                'BERTSCORE': {
                    'precision': float(
                        bertscore_result['precision'][i]
                    ),
                    'recall': float(
                        bertscore_result['recall'][i]
                    ),
                    'f1': float(bertscore_result['f1'][i]),
                },
                'PERPLEXITY': float(perplexities[i])
                if i < len(perplexities)
                else None,
            }
            results['samples'].append(sample)

        # Save to JSON file
        os.makedirs(self.output_dir, exist_ok=True)
        json_filename = os.path.join(
            self.output_dir,
            f'evaluation_results_step_{self.eval_counter}.json',
        )

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f'\nResults saved to: {json_filename}')
        print(f'ROUGE-L: {rouge_result["rougeL"]:.4f}')
        print(
            f'BERTScore F1: {np.mean(bertscore_result["f1"]):.4f}'
        )
        if avg_perplexity is not None:
            print(f'Perplexity: {avg_perplexity:.4f}')
        print('=' * 80 + '\n')

        self.eval_counter += 1

        # Return metrics for trainer
        metrics = {
            'rouge1': rouge_result['rouge1'],
            'rouge2': rouge_result['rouge2'],
            'rougeL': rouge_result['rougeL'],
            'bertscore_precision': float(
                np.mean(bertscore_result['precision'])
            ),
            'bertscore_recall': float(
                np.mean(bertscore_result['recall'])
            ),
            'bertscore_f1': float(
                np.mean(bertscore_result['f1'])
            ),
        }

        if avg_perplexity is not None:
            metrics['perplexity'] = avg_perplexity

        return metrics

    def get_trainer(self):
        return self.trainer
