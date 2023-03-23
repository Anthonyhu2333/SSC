from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset, DataLoader
import json

class Base_Eval:
    def __init__(self):
        pass
    
    @abstractmethod
    def score(self, document, claim):
        pass
    
    @abstractmethod
    def evaluate_file(self, file_path):
        pass

class SummaryDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
