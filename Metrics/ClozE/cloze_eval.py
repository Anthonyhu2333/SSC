import sys
sys.path.append('..')
import torch
from metric_utils import Base_Eval
from metric import ClozEMetric
import numpy as np
import pdb
import json

class ClozEEval(Base_Eval):
    def __init__(self):
        self.scorer = ClozEMetric(cloze_model_path='/root/autodl-tmp/SSC/Metrics/ClozE/checkpoint/ClozE-roberta-base-cnndm', fact_extractor='en_core_web_sm', use_gpu=False)
    
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
                                           use_tqdm=True)])
    
    def check_key_word(self, document, claim, keyword):
        #临时处理
        predicts = self.scorer.score([document],
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
                                    verbose=False,
                                    use_tqdm=False)
        detail = predicts[0]['infos']['summary'][0]['comparision']
        for item in detail:
            if item['factor'] == keyword:
                return item['answer']
        return keyword
    
    def check_key_words(self, document, claim):
        predicts = self.scorer.score([document],
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
                                    verbose=False,
                                    use_tqdm=False)
        detail = predicts[0]['infos']['summary'][0]['comparision']
        return [(item['factor'], item['answer']) for item in detail]
    

if __name__ == "__main__":
    eval = ClozEEval()
    result = eval.score('Bob went to Beijing', 'Bob went to Beijing')
    


        
