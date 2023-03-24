import sys
sys.path.append('..')
import torch
from metric_utils import Base_Eval
from summac.model_summac import SummaCZS
import numpy as np

class SummaCZSEval(Base_Eval):
    def __init__(self):
        self.model_ = SummaCZS(granularity="sentence", model_name="vitc", device="cuda") 
        
    def score(self, document, claim):
        return self.model([document], [summary])["scores"][0]
    
    def evaluate_file(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        documents = [item['document'] for item in data]
        claims = [item['claim'] for item in data]
        return np.mean(self.model.score(documents, claims)["scores"])