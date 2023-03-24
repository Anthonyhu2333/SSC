import sys
sys.path.append('..')
import torch
from metric_utils import Base_Eval

class FEQAEval(Base_Eval):
    def __init__(self):
        pass
    
    def score(self, docuemnt, claim):
        pass
    
    def evaluate_file(self, file_path):
        pass