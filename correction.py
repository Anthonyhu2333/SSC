import sys
sys.path.append('/root/autodl-tmp/SSC/DataGeneration')
from syntacticTree import SyntacticTrees
from Metrics.metric_utils import Base_Eval
from collections import Counter
import requests
import json
import pdb
import spacy

class CorrectionModel:
    def __init__(self, nlp=None):
        if nlp == None:
            self.spacy = spacy.load("en_core_web_sm")
        else:
            self.spacy = nlp
        self.deletoin_relation = ['acl', 'dep', 'appos', 'advcl', 'agent', 'nmod', 'advmod', 'prt', 'prep', 'amod', 'compound', 'neg', 'relcl', 'punct', 'ccomp', 'poss', 'conj', 'pcomp', 'xcomp']
        self.replace_relation = ['entity']


    def correct_ssc(self, claim, document, scorers=None, weights=None):
        if not scorers:
            scorers = Base_Eval()
        if weights and len(weights) != len(scorers):
            raise ValueError("The number of scores cannot match the number of weights")
        if not weights:
            weights = [1 for i in scorers]
        trees = SyntacticTrees(claim, nlp=self.spacy).trees
        output_list = []
        for tree in trees:
            check_list = tree.head.get_check_list()
            score = sum([scorer.score(document=document, claim=tree.get_sentence())*weight for scorer, weight in zip(scorers, weights)])
            output = tree.get_sentence()
            for item in check_list:
                if item.rel in self.deletoin_relation:       
                    index = item.head.__dict__[item.head_direction].index(item)
                    tree.delete_subtree(item)
                    result = sum([scorer.score(document=document, claim=tree.get_sentence())*weight for scorer, weight in zip(scorers, weights)])
                    print(result)
                    if result > score:
                        output = tree.get_sentence()
                        score = result
                    item.head.__dict__[item.head_direction].insert(index, item)
            output_list.append(output)                 
        return ' '.join(output_list)

class Scorer():
    def __init__(self, url):
        self.url = url
    
    def score(self, claim, document):
        # 定义请求数据
        data = {'document': document, 'claim': claim}
        # 发送POST请求并获取响应
        response = requests.post(self.url, json=data)
        # 解析响应JSON数据
        result = json.loads(response.text)
        return result

if __name__ == "__main__":
    with open('/root/autodl-tmp/SSC/frank_sentence.json', 'r') as f:
        data = json.load(f)
    c = CorrectionModel()
    cloze_score = Scorer(url='http://localhost:10006/summacconv')
    for item in data:
        document = item['document']
        claim = item['claim']
        output = c.correct_ssc(claim, document, [cloze_score])


    
            
        

        