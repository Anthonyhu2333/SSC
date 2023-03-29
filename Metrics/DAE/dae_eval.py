import sys
sys.path.append('..')
import torch
from metric_utils import Base_Eval
from preprocessing_utils import get_tokens, get_relevant_deps_and_context
from sklearn.utils.extmath import softmax
import numpy as np
from transformers import ElectraConfig, ElectraTokenizer
import utils
import subprocess
import argparse
import pdb

class DAEEval(Base_Eval):
    def __init__(self):
        # subprocess.run('cd /root/autodl-tmp/SSC/Metrics/DAE/checkpoint/stanford-corenlp-full-2018-02-27 && java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer &', shell=True, check=True)

        parser = argparse.ArgumentParser()
        self.args = parser.parse_args()
        self.args.n_gpu = 1
        device = torch.device("cuda")
        self.args.device = device
        self.args.model_type = 'electra_dae'
        self.args.dependency_type = 'enhancedDependencies'
        self.tokenizer = ElectraTokenizer.from_pretrained('/root/autodl-tmp/SSC/Metrics/DAE/checkpoint/dae_basic')
        self.model = utils.ElectraDAEModel.from_pretrained('/root/autodl-tmp/SSC/Metrics/DAE/checkpoint/dae_basic')
        self.model.to(self.args.device)
        

    def score(self, document, claim):
        return score_example_single_context(claim, document, self.model, self.tokenizer, self.args)
    
    def evaluate_file(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        documents = [item['document'] for item in data]
        claims = [item['claim'] for item in data]
        scores = []
        for document, claim in zip(documents, claims):
            scores.append(self.score(document, claim))
        return np.mean(scores)
    
    def check_key_word(self, document, claim, keyword):
        return None
    
    def check_key_words(self, document, claim):
        return None

def score_example_single_context(decode_text, input_text, model, tokenizer, args):
    gen_tok, _, gen_dep = get_relevant_deps_and_context(decode_text, args)

    tokenized_text = get_tokens(input_text)

    ex = {'input': tokenized_text, 'deps': [], 'context': ' '.join(gen_tok), 'sentlabel': -1}
    for dep in gen_dep:
        ex['deps'].append({'dep': dep['dep'], 'label': -1, 'head_idx': dep['head_idx'] - 1,
                           'child_idx': dep['child_idx'] - 1, 'child': dep['child'], 'head': dep['head']})

    dict_temp = {'id': 0, 'input': ex['input'], 'sentlabel': ex['sentlabel'], 'context': ex['context']}
    for i in range(20):
        if i < len(ex['deps']):
            dep = ex['deps'][i]
            dict_temp['dep_idx' + str(i)] = str(dep['child_idx']) + ' ' + str(dep['head_idx'])
            dict_temp['dep_words' + str(i)] = str(dep['child']) + ' ' + str(dep['head'])
            dict_temp['dep' + str(i)] = dep['dep']
            dict_temp['dep_label' + str(i)] = dep['label']
        else:
            dict_temp['dep_idx' + str(i)] = ''
            dict_temp['dep_words' + str(i)] = ''
            dict_temp['dep' + str(i)] = ''
            dict_temp['dep_label' + str(i)] = ''

    features = utils.convert_examples_to_features_bert(
        [dict_temp],
        tokenizer,
        max_length=128,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
    )

    # Convert to Tensors and build dataset
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(args.device)
    attention = torch.tensor([f.input_attention_mask for f in features], dtype=torch.long).to(args.device)
    token_ids = torch.tensor([f.token_ids for f in features], dtype=torch.long).to(args.device)

    child = torch.tensor([f.child_indices for f in features], dtype=torch.long).to(args.device)
    head = torch.tensor([f.head_indices for f in features], dtype=torch.long).to(args.device)

    dep_labels = torch.tensor([f.dep_labels for f in features], dtype=torch.long).to(args.device)
    num_dependencies = torch.tensor([f.num_dependencies for f in features], dtype=torch.long).to(args.device)
    arcs = torch.tensor([f.arcs for f in features], dtype=torch.long).to(args.device)
    arc_labels = torch.tensor([f.arc_labels for f in features], dtype=torch.long).to(args.device)
    arc_label_lengths = torch.tensor([f.arc_label_lengths for f in features], dtype=torch.long).to(args.device)

    inputs = {'input_ids': input_ids, 'attention': attention, 'token_ids': token_ids, 'child': child, 'head': head,
              'dep_labels': dep_labels, 'arcs': arc_labels, 'arc_label_lengths': arc_label_lengths,
              'device': args.device}

    outputs = model(**inputs)
    tmp_eval_loss, logits = outputs[:2]
    preds = logits.detach().cpu().numpy()

    # f_out = open('test.txt', 'a')
    text = tokenizer.decode(input_ids[0])
    text = text.replace(tokenizer.pad_token, '').strip()
    # f_out.write(text + '\n')
    for j, arc in enumerate(arcs[0]):
        arc_text = tokenizer.decode(arc)
        arc_text = arc_text.replace(tokenizer.pad_token, '').strip()
        if arc_text == '':
            break
        pred_temp = softmax([preds[0][j]])
        # f_out.write(arc_text + '\n')
        # f_out.write('pred:\t' + str(np.argmax(pred_temp)) + '\n')
        # f_out.write(str(pred_temp[0][0]) + '\t' + str(pred_temp[0][1]) + '\n')
        # f_out.write('\n')
    # f_out.close()
    preds = preds.reshape(-1, 2)
    preds = softmax(preds)
    preds = preds[:, 1]
    preds = preds[:num_dependencies[0]]
    score = np.mean(preds)
    return score


if __name__ == '__main__':
    eval = DAEEval()
    result = eval.score('Bob went to Beijing.', 'Bob went to Beijing.')
    pdb.set_trace()