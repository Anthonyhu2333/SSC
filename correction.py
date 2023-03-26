from SyntacticTree import SyntacticTrees
from Metrics.metric_utils import Base_Eval

class CorrectionModel:
    def __init__(self, nlp=None):
        if nlp == None:
            self.spacy = spacy.load("en_core_web_sm")
        else:
            self.spacy = nlp
        self.deletoin_relation = []
        self.replace_relation = ['entity']


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
                    inter_sentence = tree.get_sentence(item)
                    score = self.correct_node(item, tree, document, scorers, weights, score)
        return ' '.join([tree.get_sentence() for tree in trees])

    def correct_node(self, node, tree, document, scorers, weights, scoree):
        if node.rel not in self.deletoin_relation:
            return
        if not weights:
            weights = [1 for _ in scorers]
        else:
            claim = tree.get_sentence(node)
            result = sum([scorer.score(document, claim)*weight for scorer, weight in zip(scorers, weights)])
            if result <= score:
                tree.delete_subtree(node)
            else:
                score = result
        return score

        