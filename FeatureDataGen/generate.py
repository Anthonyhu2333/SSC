import sys
sys.path.append('..')
sys.path.append('/root/autodl-tmp/SSC/DataGeneration')
sys.path.append('/root/autodl-tmp/SSC/SyntacticTree')
from DataGeneration.benchmark import SummaCBenchmark, load_dataset
from SyntacticTree.syntacticTree import SyntacticTrees
from nltk import sent_tokenize
import json
import random
from tqdm import tqdm
import spacy
import pdb
spacy.prefer_gpu(0)
nlp = spacy.load("en_core_web_sm")


if __name__ == "__main__":
    benchmark = SummaCBenchmark()
    benchmark.xsum = load_dataset("xsum")["test"]
    relation_list = ['acl', 'dep', 'appos', 'advcl', 'agent', 'nmod', 'advmod', 'prt', 'prep', 'amod', 'compound', 'neg', 'relcl', 'punct', 'ccomp', 'poss', 'conj', 'pcomp', 'xcomp']
    for relation in tqdm(relation_list):
        result = []
        count = 0
        for item in tqdm(benchmark.xsum):
            summary = item['summary'].replace('\n', '')
            document_list = sent_tokenize(item['document'])
            random.shuffle(document_list)          
            try:
                tree = SyntacticTrees(summary, nlp=nlp).trees[0]
                
                summary = tree.get_sentence()
                check_list = [item for item in tree.head.get_check_list() if item.rel == relation]
                if len(check_list) == 0:
                    fake_summary = summary
                else:         
                    modify_node = random.choice(check_list)
                    head_node = modify_node.head
                    head_direction = modify_node.head_direction
                    head_index = head_node.__dict__[head_direction].index(modify_node)
                    tree.delete_subtree(modify_node)
                    for sentence in document_list:
                        sent_tree = SyntacticTrees(sentence, nlp=nlp).trees[0]
                        sent_check_list = [item for item in sent_tree.head.get_check_list() if item.rel == relation]
                        if len(sent_check_list) != 0:
                            append_node = random.choice(sent_check_list)
                            tree.add_subtree(append_node, head_direction, head_node, index=head_index)
                            count += 1
                            break
                    fake_summary = tree.get_sentence()
            except Exception as e:
                fake_summary = summary
            result.append({
                    'document': item['document'],
                    'summary': summary,
                    'fake_summary': fake_summary
                })
        output_file = open('/root/autodl-tmp/SSC/data/fake_data/xsum_'+relation+'_'+str(count)+'.json', 'w')
        output_file.writelines(json.dumps(result))
