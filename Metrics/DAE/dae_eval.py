import sys
sys.path.append('..')
import torch
from metric_utils import Base_Eval

class DEAEval(Base_Eval):
    def __init__(self):
        pass
    
    def score(self, document, claim):
        return 1
    
    def evaluate_file(self, file_path):
        return 1
    
    def check_key_word(self, document, claim, keyword):
        return None
    
    def check_key_words(self, document, claim):
        return None