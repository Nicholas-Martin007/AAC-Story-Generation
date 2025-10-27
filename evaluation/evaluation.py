import json
import os
import sys

import torch
from datasets import load_from_disk
from evaluate import load
from peft import PeftModelForCausalLM, PeftModelForSeq2SeqLM
from tqdm import tqdm

sys.path.append(os.path.abspath('./'))
from config import *
from fine_tuning.f_tokenizer import FinetuneTokenizer


class Evaluate:
    def __init__(
        self,
        model_path,
        model_type,
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.tokenizer = FinetuneTokenizer(
            model_path
        ).get_tokenizer()

        self.rouge = load('rouge')
        self.bertscore = load('bertscore')

    def load_model(self, checkpoint_path):
        """
        Load Model (SEQ2SEQ atau CAUSAL)
        """

        print(f'Path: {checkpoint_path}...')

        if self.model_type == 'seq2seq':
            from transformers import AutoModelForSeq2SeqLM

            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                device_map=DEVICE,
            )

            model.resize_token_embeddings(len(self.tokenizer))

            qlora_model = PeftModelForSeq2SeqLM.from_pretrained(
                model,
                checkpoint_path,
                device_map=DEVICE,
                inference_mode=False,
            )
        else:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=DEVICE,
            )
            model.resize_token_embeddings(len(self.tokenizer))
            qlora_model = PeftModelForCausalLM.from_pretrained(
                model,
                checkpoint_path,
                device_map=DEVICE,
                inference_mode=False,
            )

        model.resize_token_embeddings(len(self.tokenizer))

        merged_model = qlora_model.merge_and_unload()
        merged_model = merged_model.to(DEVICE)
        merged_model.eval()  # change to eval mode

        return merged_model

    def clean_causal_generated_text(self, generated_text):
        patterns = [
            r'User:.*?(?=Assistant:|$)',
            r'Assistant:\s*',
            r'\[.*?\]',
            r'<\|.*?\|>',
            r'###.*?(?=\n|$)',
        ]

        import re

        result = generated_text
        for pattern in patterns:
            result = re.sub(pattern, '', result, flags=re.DOTALL)

        result = re.sub(r'\n+', '\n', result)
        result = result.strip()

        return result

    def generate_text(
        self,
        model,
        input,
        max_new_tokens,
    ):
        """INFERENCE"""
        if self.model_type == 'causal':
            prompt = self.tokenizer.apply_chat_template(
                [
                    {
                        'role': 'user',
                        'content': json.dumps(input),
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = json.dumps(input)

        input_ids = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        ).input_ids.to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        if self.model_type == 'causal':
            input_length = input_ids.shape[1]
        else:
            input_length = 0

        generated_tokens = output.sequences[:, input_length:]
        generated_text = self.tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True
        )

        if self.model_type == 'causal':
            generated_text = self.clean_causal_generated_text(
                generated_text=generated_text
            )

        return {'generated_text': generated_text}

    def calculate_perplexity(self, model, text):
        tokenized_text = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
        ).to(DEVICE)
        with torch.no_grad():
            loss = model(
                **tokenized_text,
                labels=tokenized_text.input_ids,
            ).loss

        return torch.exp(loss).item()

    def calculate_scores(self, predictions, references):
        """
        EVALUATE ROUGE / BERT
        """

        rouge = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_aggregator=True,
        )
        bert = self.bertscore.compute(
            predictions=predictions,
            references=references,
            lang='id',
            model_type='bert-base-multilingual-cased',
        )

        return {
            'rouge': rouge,
            'bertscore': {
                'precision': sum(bert['precision'])
                / len(bert['precision']),
                'recall': sum(bert['recall'])
                / len(bert['recall']),
                'f1': sum(bert['f1']) / len(bert['f1']),
            },
        }

    def save_model(self, results, scores, fine_tuned_model_name):
        # SAVE GENERATIONS
        os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)
        with open(
            os.path.join(
                EVALUATION_OUTPUT_DIR,
                f'{fine_tuned_model_name}_generations.json',
            ),
            'w',
            encoding='utf-8',
        ) as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # SAVE SCORES
        with open(
            os.path.join(
                EVALUATION_OUTPUT_DIR,
                f'{fine_tuned_model_name}_scores.json',
            ),
            'w',
            encoding='utf-8',
        ) as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)

    def evaluate_model(
        self,
        fine_tuned_model_path,
        fine_tuned_model_name,
        test_data,
    ):
        model = self.load_model(fine_tuned_model_path)

        predictions, references, perplexities, results = (
            [],
            [],
            [],
            [],
        )

        # buat track
        for i, row in enumerate(tqdm(test_data)):
            input_prompt = json.loads(
                row['messages'][1]['content']
            )
            reference = row['messages'][2]['content']

            out = self.generate_text(
                model=model,
                input=input_prompt,
                max_new_tokens=512,
            )
            generated_text = out['generated_text']

            # Perplexity per data
            perplexity = self.calculate_perplexity(
                model, generated_text
            )

            predictions.append(generated_text)
            references.append(reference)

            perplexities.append(perplexity)

            results.append(
                {
                    'index': i,
                    'input': input_prompt,
                    'reference': reference,
                    'generated': generated_text,
                    'perplexity': perplexity,
                }
            )

        # CALCULATE ROUGE / BERTSCORE
        scores = self.calculate_scores(predictions, references)

        # CALCULATE PERPLEXITY
        scores['perplexity'] = {
            'mean': sum(perplexities) / len(perplexities),
            'min': min(perplexities),
            'max': max(perplexities),
        }

        self.save_model(
            results=results,
            scores=scores,
            fine_tuned_model_name=fine_tuned_model_name,
        )

        del model
        torch.cuda.empty_cache()
        return scores


if __name__ == '__main__':
    dataset = load_from_disk('./hf_dataset_oktober_13')
    dataset_split = dataset.train_test_split(
        test_size=0.1, seed=SEED
    )

    test_data = dataset_split['test']
    test_data = test_data.select(range(5))

    for name, data in FINE_TUNED_MODELS.items():
        print(f'\nModel: {name}')
        eval = Evaluate(
            model_path=data['path'],
            model_type=data['model_type'],
        )
        for fine_tuned_path in data['qlora_model']:
            fine_tuned_model_name = os.path.basename(
                fine_tuned_path.rstrip('/')
            )

            eval.evaluate_model(
                fine_tuned_model_name=fine_tuned_model_name,
                fine_tuned_model_path=fine_tuned_path,
                test_data=test_data,
            )
