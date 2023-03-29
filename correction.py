from SyntacticTree import SyntacticTrees
from Metrics.metric_utils import Base_Eval
from collections import Counter

class CorrectionModel:
    def __init__(self, nlp=None):
        if nlp == None:
            self.spacy = spacy.load("en_core_web_sm")
        else:
            self.spacy = nlp
        self.deletoin_relation = []
        self.replace_relation = ['entity']


    def correct_ssc(self, claim, document, scorers=None, weights=None):
        if not scorers:
            scorers = Base_Eval()
        if weights and len(weights) != len(scorers):
            raise ValueError("The number of scores cannot match the number of weights")
        trees = SyntacticTrees(claim, nlp=self.spacy)
        output_list = []
        for tree in trees:
            check_list = tree.head.get_check_list()
            score = sum([scorer.score(document, tree.get_sentence())*weight for scorer, weight in zip(scorers, weights)])
            output = tree.get_sentence()
            for item in check_list:
                if item.rel in self.deletoin_relation: 
                    index = node.head.__dict__[node.head_direction].index(node)
                    tree.delete_subtree(item)
                    result = sum([scorer.score(document, tree.get_sentence())*weight for scorer, weight in zip(scorers, weights)])
                    if result > score:
                        output = tree.get_sentence()
                        score = result
                    item.head.__dict__[item.head_direction].insert(index, item)
                output_list.append(output)                 
        return ' '.join(output_list)
    
    
    def correct_rssc(self, claim, document, scorers=None, weights=None):
        if not scorers:
            scorers = Base_Eval()
        if weights and len(weights) != len(scorers):
            raise ValueError("The number of scores cannot match the number of weights")
        trees = SyntacticTrees(claim, nlp=self.spacy)
        for tree in trees:
            check_list = tree.head.get_check_list()
            score = 0
            for item in check_list:
                if item.rel in self.deletoin_relation or item.rel in self.replace_relation:
                    self.correct_node(item, tree, document, scorers, weights, score)
        return ' '.join([tree.get_sentence() for tree in trees])

    def correct_node(self, node, tree, document, scorers, weights, scoree):
        if not weights:
            weights = [1 for _ in scorers]
        if node.rel in self.deletoin_relation:
            score_1 = sum([scorer.score(document, tree.get_sentence(node))*weight for scorer, weight in zip(scorers, weights)])
            index = node.head.__dict__[node.head_direction].index(node)
            tree.delete_subtree(node)
            score_2 = sum([scorer.score(document, tree.get_sentence())*weight for scorer, weight in zip(scorers, weights)])
            if score_1 >= score_2:
                node.head.__dict__[node.head_direction].insert(index, node)

        if node.rel in self.deletoin_relation:
            claim = tree.get_sentence(node)
            keyword = node.text
            candidate_list = [scorer.check_key_word(docuemnt, claim, keyword) for scorer in scorers]
            candidate_list = [item for item in candidate_list if item is not None]
            if candidate_list:
                candidate_word = Counter(candidate_list).most_common(1)[0][0]
                claim_1 = tree.get_sentence(node)
                original_text = node.text
                tree.replace_subtree_node(node, candidate_word)
                claim_2 = tree.get_sentence(node)
                score_1 = sum([scorer.score(document, claim_1)*weight for scorer, weight in zip(scorers, weights)])
                score_2 = sum([scorer.score(document, claim_2)*weight for scorer, weight in zip(scorers, weights)])
                if score_1 >= score_2:
                    tree.replace_subtree_node(node, original_text)

            
        

        