import sys
sys.path.append('..')
from metric_utils import Base_Eval
from transformers import pipeline
import numpy as np
import json
from collections import Counter
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import pdb

class ColaEval(Base_Eval):
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained("/root/autodl-tmp/SSC/Metrics/CoLA/checkpoint/cola_model")
        self.tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/SSC/Metrics/CoLA/checkpoint/cola_model")
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
       
    def score(self, document=None, claim=None):
        result = self.classifier(claim)
        if result['label'] == 'acceptable':
            return 1
        else:
            return 0

    
    
    def evaluate_file(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        claims = [item['claim'] for item in data]
        result = self.classifier(claims, batched=True)
        return Counter([item['label'] for item in result])['acceptable']/len(result)

        
if __name__ == "__main__":
    eval = ColaEval()
    result = eval.evaluate_file('/root/autodl-tmp/SSC/Metrics/test.json')
    