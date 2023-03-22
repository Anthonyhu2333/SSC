from abc import ABC, abstractmethod

class Base_Eval:
    def __init__(self):
        pass
    
    @abstractmethod
    def score(self, document, claim):
        pass
    
    @abstractmethod
    def evaluate_file(self, file_path):
        pass