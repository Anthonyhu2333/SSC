import sys
sys.path.append('..')
import torch
from base_eval import Base_Eval
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np
import pdb

class FactccEval(Base_Eval):
    def __init__(self):
        self.checkpoint = '/root/autodl-tmp/SSC/Metrics/FactCC/checkpoint/factcc-checkpoint'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = BertForSequenceClassification.from_pretrained(self.checkpoint)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model.to(self.device)
        
    def score(self, document, claim):
        encoded_input = self.tokenizer.encode_plus(claim, document, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        output = self.model(**encoded_input)
        preds = np.argmax(output.logits.cpu().detach().numpy(), axis=1)
        return int(preds[0])
        pdb.set_trace()


    def evaluate_file(self, file_path):
        
        
        pass

if __name__ == "__main__":
    model = FactccEval()
    model.score('Delta compression using up to 128 threads.','Delta compression using up to 128 threads.')



