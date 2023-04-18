import sys
sys.path.append('/root/autodl-tmp/SSC/DataGeneration')
from syntacticTree import SyntacticTrees
from Metrics.metric_utils import Base_Eval
from collections import Counter
import requests
import json
import pdb
import spacy
from tqdm import tqdm

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
                    if result > score:
                        output = tree.get_sentence()
                        score = result
                    item.head.__dict__[item.head_direction].insert(index, item)
            output_list.append(output)                 
        return ' '.join(output_list)

class Scorer():
    def __init__(self, url, name='scorer'):
        self.url = url
        self.name = name
    
    def score(self, claim, document):
        # 定义请求数据
        data = {'document': document, 'claim': claim}
        # 发送POST请求并获取响应
        response = requests.post(self.url, json=data)
        # 解析响应JSON数据
        result = json.loads(response.text)
        return result

    url_dic = {
        'cloze': 10000,
        'dae': 10002,
        'factcc': 10003,
        'feqa': 10004,
        'summacconv': 10006,
    }
    url_pre = 'http://localhost:'

if __name__ == "__main__":
    with open('/root/autodl-tmp/SSC/frank_sentence.json', 'r') as f:
        data = json.load(f)
    c = CorrectionModel()
    cloze_scorer = Scorer(url='http://localhost:10000/cloze', name='cloze')
    dae_scorer = Scorer(url='http://localhost:10002/dae', name='dae')
    factcc_scorer = Scorer(url='http://localhost:10003/factcc', name='factcc')
    feqa_scorer = Scorer(url='http://localhost:10004/feqa', name='feqa')
    quals_scorer = Scorer(url='http://localhost:10005/quals', name='quals')
    summacconv_scorer = Scorer(url='http://localhost:10006/summacconv', name='summacconv')

    scorer_list = [quals_scorer]
    
    for item in tqdm(data):
        document = item['document']
        claim = item['claim']
        try:
            output = c.correct_ssc(claim, document, scorer_list)
        except Exception as e:
            output = claim
        item['corrected_claim'] = output
    
    metric_name = '_'.join([item.name for item in scorer_list])
    with open('/root/autodl-tmp/SSC/data/correction_result/'+ metric_name+'.json', 'w') as f:
        f.writelines(json.dumps(data))


    
            
        

        