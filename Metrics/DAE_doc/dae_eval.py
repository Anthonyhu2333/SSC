# -*- coding:utf-8 -*-
# author Li Yiyang
# Modify the DAE 'evaluate_generated_outputs.py' script to DAEEvaluator.

import torch
import numpy as np
import argparse

from tqdm import tqdm
from nltk import sent_tokenize
from pycorenlp import StanfordCoreNLP
from train import MODEL_CLASSES
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from train_utils import get_single_features
from sklearn.utils.extmath import softmax


"""
python3 evaluate_generated_outputs.py \
        --model_type electra_dae \
        --model_dir $MODEL_DIR  \
        --input_file sample_test.txt
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='electra_dae', type=str)
    parser.add_argument("--model_dir", default='ENT-C_dae', type=str)
    parser.add_argument("--max_seq_length", default=512)
    parser.add_argument("--input_file", type=str, required=False, )
    parser.add_argument("--gpu_device", type=int, default=0, help="gpu device")
    args = parser.parse_args()
    return args


class DAEEvaluator:
    def __init__(self) -> None:
        
        self.args = get_args()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nlp = StanfordCoreNLP('http://localhost:9000')

        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.args.model_type]
        self.tokenizer = tokenizer_class.from_pretrained(self.args.model_dir)
        self.model = model_class.from_pretrained(self.args.model_dir).to(self.device)
        self.model.eval()
    
    def evaluate_summary(self, article_data, summary):

        eval_dataset = get_single_features(summary, article_data, self.tokenizer, self.nlp, self.args)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

        # 一些时候，数据会处理出错，目前暂且扔掉这些数据
        # 这种情况下应该是没有弧（arc）的情况
        if len(eval_dataloader) < 1:
            return None

        batch = [t for t in eval_dataloader][0]

        batch = tuple(t.to(self.device) for t in batch)

        with torch.no_grad():
            input_ids, attention, child, head = batch[0], batch[1], batch[2], batch[3]
            mask_entail, mask_cont, num_dependency, arcs = batch[4], batch[5], batch[6], batch[7]
            sent_labels = batch[8]

            inputs = {'input_ids': input_ids, 'attention': attention, 'child': child,
                    'head': head, 'mask_entail': mask_entail, 'mask_cont': mask_cont,
                    'num_dependency': num_dependency, 'sent_label': sent_labels, 'device': self.device}

            outputs = self.model(**inputs)
            dep_outputs = outputs[1].detach()
            dep_outputs = dep_outputs.squeeze(0)
            dep_outputs = dep_outputs[:num_dependency, :].cpu().numpy()

            input_full = self.tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=False)
            input_full = ' '.join(input_full).replace('[PAD]', '').strip()

            summary = input_full.split('[SEP]')[1].strip()
            
            predictions = []

            for j, arc in enumerate(arcs[0]):
                arc_text = self.tokenizer.decode(arc)
                arc_text = arc_text.replace(self.tokenizer.pad_token, '').strip()

                if arc_text == '':  # for bert
                    break

                softmax_probs = softmax([dep_outputs[j]])
                pred = np.argmax(softmax_probs[0])

                predictions.append(pred.item())

            return predictions

    def score_one_sample(self, document, sentence):
        return self.evaluate_summary(document.lower(), sentence.lower())

    def score(self, documents, summaries, use_tqdm=True):
        
        # 参照repo中的描述
        # EDIT: For running models on non-preprocessed data, the input file needs to be preprecessed in the following way:
        #   Run both input article and summary through PTB tokenizer.   （本脚本未考虑）
        #   Lower case both input article and summary.  （已考虑）
        # The models expect input of the above form. Not pre-processing it appropriately will hurt model performance.

        documents = [document for document in documents]
        summaries = [summary for summary in summaries]

        scores = []
        bar = tqdm(list(zip(documents, summaries)), desc='Evaluating', ncols=150) if use_tqdm else zip(documents, summaries)

        for document, summary in bar:
            results = []

            for sentence in sent_tokenize(summary):
                predictions = self.evaluate_summary(document.lower(), sentence.lower())
                if predictions is not None:
                    results.extend(predictions)

            scores.append(1 if len(results)==0 else sum(results)/len(results))
        
        return scores
