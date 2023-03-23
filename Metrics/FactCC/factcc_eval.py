import sys
sys.path.append('..')
import torch
from metric_utils import Base_Eval
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import pdb

class SummaryDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]['claim'] + ' [SEP] ' + self.data[index]['document']

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


    def evaluate_file(self, file_path):
        mydataset = SummaryDataset(file_path)
        dataloader = DataLoader(mydataset, batch_size=4)
        for batch in dataloader:
            encoded_input = self.tokenizer.batch_encode_plus(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
            output = self.model(**encoded_input)
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        return np.mean(preds)
        
        

if __name__ == "__main__":
    model = FactccEval()
    model.evaluate_file('/root/autodl-tmp/SSC/Metrics/test.json')




