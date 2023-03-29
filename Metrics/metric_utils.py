import torch
from torch.utils.data import Dataset, DataLoader
import json

class Base_Eval:
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


class SummaryDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
