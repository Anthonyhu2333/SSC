import sys
sys.path.append('..')
import torch
from metric_utils import Base_Eval
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from tqdm import tqdm
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
        self.tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/SSC/Metrics/FactCC/checkpoint/tokenizer')
        self.model.to(self.device)
        
    def score(self, document, claim):
        encoded_input = self.tokenizer.encode_plus(claim, document, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        output = self.model(**encoded_input)
        preds = np.argmax(output.logits.cpu().detach().numpy(), axis=1)
        return int(preds[0])


    def evaluate_file(self, file_path):
        mydataset = SummaryDataset(file_path)
        dataloader = DataLoader(mydataset, batch_size=4)
        result = None
        for batch in tqdm(dataloader):
            encoded_input = self.tokenizer.batch_encode_plus(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
            output = self.model(**encoded_input)
            preds = np.argmax(output.logits.cpu().detach().numpy(), axis=1)
            if result is None:
                result = preds
            else:
                result = np.append(result, preds, axis=0)
        if result is None:
            return None
        return np.mean(result)
    
    def check_key_word(self, document, claim, keyword):
        return None
    
    def check_key_words(self, document, claim):
        return None
        
        

if __name__ == "__main__":
    model = FactccEval()
    a = model.evaluate_file('/root/autodl-tmp/SSC/Metrics/test.json')
    print(a)




