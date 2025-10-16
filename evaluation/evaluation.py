import os
import sys
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Any
from datasets import load_from_disk
from evaluate import load
from peft import PeftModelForCausalLM, PeftModelForSeq2SeqLM
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)


sys.path.append(os.path.abspath('./'))
from config import *
from fine_tuning.f_tokenizer import FinetuneTokenizer
from fine_tuning.f_model import FinetuneModel
from fine_tuning.f_lora import FinetuneLora


class Evaluate:
    def __init__(
        self,
        base_model_path: str,
        model_type: str,
        device: str,
    ):
        self.base_model_path = base_model_path
        self.model_type = model_type
        self.device = device

        self.tokenizer = FinetuneTokenizer(
            base_model_path
        ).get_tokenizer()

        self.rouge = load('rouge')
        self.bertscore = load('bertscore')

    def load_model(self, checkpoint_path: str):
        print(f'Loading model from {checkpoint_path}...')

        if self.model_type == 'seq2seq':
            from transformers import AutoModelForSeq2SeqLM

            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model_path,
                device_map=self.device,
            )

            base_model.resize_token_embeddings(
                len(self.tokenizer)
            )
            qlora_model = PeftModelForSeq2SeqLM.from_pretrained(
                base_model,
                checkpoint_path,
                device_map=self.device,
                inference_mode=False,
            )
        else:
            from transformers import AutoModelForCausalLM

            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                device_map=self.device,
            )
            qlora_model = PeftModelForCausalLM.from_pretrained(
                base_model,
                checkpoint_path,
                device_map=self.device,
                inference_mode=False,
            )

        base_model.resize_token_embeddings(len(self.tokenizer))
        merged_model = qlora_model.merge_and_unload()
        merged_model = merged_model.to(self.device)
        merged_model.eval()

        return merged_model

    def generate_text(
        self,
        model,
        input_data: List[str],
        max_new_tokens: int = 512,
    ):
        if self.model_type == 'causal':
            prompt = self.tokenizer.apply_chat_template(
                [
                    {
                        'role': 'user',
                        'content': json.dumps(input_data),
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = json.dumps(input_data)

        input_ids = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        ).input_ids.to(self.device)
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

        input_len = (
            input_ids.shape[1]
            if self.model_type == 'causal'
            else 0
        )
        generated_tokens = output.sequences[:, input_len:]
        generated_text = self.tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True
        )

        return {'generated_text': generated_text}

    def calculate_perplexity(self, model, text: str) -> float:
        enc = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            loss = model(**enc, labels=enc.input_ids).loss
        return torch.exp(loss).item()

    def calculate_scores(
        self, preds: List[str], refs: List[str]
    ) -> Dict[str, Any]:
        rouge = self.rouge.compute(
            predictions=preds,
            references=refs,
            use_aggregator=True,
        )
        # bert = self.bertscore.compute(
        #     predictions=preds,
        #     references=refs,
        #     lang='id',
        #     model_type='bert-base-multilingual-cased',
        # )
        return {
            'rouge': rouge,
            # 'bertscore': {
            #     'precision': sum(bert['precision'])
            #     / len(bert['precision']),
            #     'recall': sum(bert['recall'])
            #     / len(bert['recall']),
            #     'f1': sum(bert['f1']) / len(bert['f1']),
            # },
        }

    def evaluate_model(
        self,
        checkpoint_path,
        checkpoint_name,
        test_data,
        output_dir,
    ):
        os.makedirs(output_dir, exist_ok=True)
        model = self.load_model(checkpoint_path)

        preds, refs, perplexities, results = [], [], [], []

        for idx, row in enumerate(
            tqdm(test_data.select(range(1)))
        ):
            input_data = json.loads(
                row['messages'][1]['content']
            )
            reference = row['messages'][2]['content']
            out = self.generate_text(model, input_data)
            gen = out['generated_text']
            ppl = self.calculate_perplexity(model, gen)

            preds.append(gen)
            refs.append(reference)
            perplexities.append(ppl)

            results.append(
                {
                    'index': idx,
                    'input': input_data,
                    'reference': reference,
                    'generated': gen,
                    'perplexity': ppl,
                }
            )

        with open(
            os.path.join(
                output_dir, f'{checkpoint_name}_generations.json'
            ),
            'w',
            encoding='utf-8',
        ) as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        scores = self.calculate_scores(preds, refs)
        scores['perplexity'] = {
            'mean': sum(perplexities) / len(perplexities),
            'min': min(perplexities),
            'max': max(perplexities),
        }

        with open(
            os.path.join(
                output_dir, f'{checkpoint_name}_scores.json'
            ),
            'w',
            encoding='utf-8',
        ) as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)

        del model
        torch.cuda.empty_cache()
        return scores


def main():
    OUTPUT_DIR = './evaluation_results'

    dataset = load_from_disk('./hf_dataset_oktober_13')
    dataset_split = dataset.train_test_split(
        test_size=0.1, seed=42
    )
    test_data = dataset_split['test']

    MODELS = {
        # 'llama': {
        #     'path': MODEL_PATH['llama3.2-3b'],
        #     'model_type': 'causal',
        #     'qlora_model': [
        #         '/Users/Nicmar/Documents/coding/LLM QLORA/llama_downloads/llama3_2-3b_lr1_4421754150410843e-05_wd0_02_r48_a96_ep1_bs1',
        #     ],
        # },
        # 'mistral': {
        #     'path': MODEL_PATH['mistral7b'],
        #     'model_type': 'causal',
        #     'qlora_model': [
        #         '/Users/Nicmar/Documents/coding/LLM QLORA/mistral_downloads/mistral7b_lr0_0003159894127911858_wd0_03_r48_a96_ep2_bs4',
        #     ],
        # },
        'flan': {
            'path': MODEL_PATH['flan-large'],
            'model_type': 'seq2seq',
            'qlora_model': [
                '/Users/Nicmar/Documents/coding/LLM QLORA/flan-t5_downloads/flan-large_lr0_0001482178201997769_wd0_09_r32_a16_ep1_bs4',
            ],
        },
    }

    for name, data in MODELS.items():
        print(f'\nModel: {name}\n' + '=' * 80)
        evaluator = Evaluate(
            data['path'], data['model_type'], DEVICE
        )
        for idx, ckpt in enumerate(data['qlora_model'], 1):
            ckpt_name = f'{name}_checkpoint_{idx}'
            print(
                f'\nEvaluating {ckpt_name} ({idx}/{len(data["qlora_model"])})'
            )
            scores = evaluator.evaluate_model(
                ckpt, ckpt_name, test_data, OUTPUT_DIR
            )
            print(f'ROUGE-1: {scores["rouge"]["rouge1"]:.4f}')

            print(
                f'Perplexity: {scores["perplexity"]["mean"]:.4f}'
            )


if __name__ == '__main__':
    main()
