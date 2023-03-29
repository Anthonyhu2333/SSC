import sys
sys.path.append('..')
import torch
from metric_utils import Base_Eval
from summac.model_summac import SummaCConv
import numpy as np
import pdb

class SummaCConvEval(Base_Eval):
    def __init__(self):
        self.model = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")

    def score(self, document, claim):
        result = self.model.score([document], [claim])['scores'][0]
        return result
    
    def evaluate_file(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        documents = [item['document'] for item in data]
        claims = [item['claim'] for item in data]
        return np.mean(self.model.score(documents, claims)["scores"])
    
    def check_key_word(self, document, claim, keyword):
        return None
    
    def check_key_words(self, document, claim):
        return None