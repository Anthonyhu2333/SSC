import sys
sys.path.append('..')
import torch
from metric_utils import Base_Eval
from metric import ClozEMetric
import numpy as np

class ClozEEval(Base_Eval):
    def __init__(self):
        self.scorer = ClozEMetric(cloze_model_path='/root/autodl-tmp/SSC/Metrics/ClozE/checkpoint/ClozE-roberta-base-cnndm', fact_extractor='en_core_web_sm', use_gpu=True)
    
    def score(self, document, claim):
        return self.scorer.score([document],
                                           [claim],
                                           k=1,
                                           selection='entity_first',
                                           granularity='sentence',
                                           criterion='f1_score',
                                           use_confidence=True,
                                           use_sentencizer=False,
                                           summary_strategy='average',
                                           sentence_strategy='average',
                                           confidence_strategy='average',
                                           alpha=0.5,
                                           beta=0.5,
                                           eval_batch_size=8,
                                           verbose=True,
                                           use_tqdm=False)[0]['score']
    
    def evaluate_file(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        documents = [item['document'] for item in data]
        claims = [item['claim'] for item in data]
        return np.mean([item['score'] for item in self.scorer.score(documents,
                                           claims,
                                           k=1,
                                           selection='entity_first',
                                           granularity='sentence',
                                           criterion='f1_score',
                                           use_confidence=True,
                                           use_sentencizer=False,
                                           summary_strategy='average',
                                           sentence_strategy='average',
                                           confidence_strategy='average',
                                           alpha=0.5,
                                           beta=0.5,
                                           eval_batch_size=8,
                                           verbose=True,
                                           use_tqdm=False)])
    
    def check_key_word(self, document, claim, key_word):
        pass
    
    def check_key_words(self, document, claim, key_word):
        pass

        
