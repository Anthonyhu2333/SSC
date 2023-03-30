import sys
sys.path.append('..')
from metric_utils import Base_Eval
# import importlib.util
# module_path = "/root/autodl-tmp/SSC/Metrics/QUALS/fairseq/models/bart/model.py"
# spec = importlib.util.spec_from_file_location("BARTModel", module_path)
# module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(module)
from bart import BARTModel
from fairseq.data import LanguagePairDataset
from fairseq.sequence_scorer import SequenceScorer
from fairseq import utils
from collections import OrderedDict
from tqdm import tqdm
import torch
import numpy as np
import pdb

class QUALSEval(Base_Eval):
    def __init__(self):
        checkpoint_dir = '/root/autodl-tmp/SSC/Metrics/QUALS/checkpoint'
        ckp_file = 'checkpoint2.pt'
        self.bart = BARTModel.from_pretrained(checkpoint_dir, checkpoint_file=ckp_file) 
        self.bart.cuda()
        self.bart.eval()
        self.bart.half()

        self.beam = 60
        self.max_len = 60
        self.min_len = 8
        self.sampling = False
        self.sampling_topk = -1
        self.sampling_topp = -1.0
        self.return_all = True
        self.diverse_beam_groups = 60
        self.diverse_beam_strength = 0.5


    def _sample_wrapper(self, sentences):
        model = self.bart
        hypotheses_batch, score_batch, unnormalized_score_batch= model.sample(
            sentences=sentences,
            beam=self.beam,
            lenpen=1.0,
            max_len_b=self.max_len,
            min_len=self.min_len,
            sampling=self.sampling,
            sampling_topk=self.sampling_topk,
            sampling_topp=self.sampling_topp,
            return_all=self.return_all,
            input_is_bpe=False,
            diverse_beam_groups=self.diverse_beam_groups,
            diverse_beam_strength=self.diverse_beam_strength,
        )
        return hypotheses_batch, score_batch, unnormalized_score_batch

    def filter_qas_dataset_lm_score(self, hypotheses, scores, unnormalized_scores, text):
        filtered_list = []
        for qa, score, unnormalized_score in zip(hypotheses, scores, unnormalized_scores):
             q_a_split = qa.split(' strutConnector')
             if len(q_a_split) == 2 and q_a_split[1].lower() in text.lower():
                filtered_list.append((q_a_split[0], q_a_split[1], score, unnormalized_score))
        filtered_list = sorted(filtered_list, key=lambda t: -t[3])
        seen_ans_dict = OrderedDict()
        for tmp in filtered_list:
            if tmp[1].lower() not in seen_ans_dict:
                seen_ans_dict[tmp[1].lower()] = [tmp,]
            else:
                seen_ans_dict[tmp[1].lower()].append(tmp)
        filtered_qa_dict = []
        max_qas = 10
        keep_adding = True
        ans_question_set_dict = {}
        while keep_adding and len(filtered_qa_dict) < max_qas:
            keep_adding = False
            for key, value in seen_ans_dict.items():
                 if value:
                    tmp = value.pop(0)
                    q, a, ns, uns = tmp[0], tmp[1], tmp[2], tmp[3]
                    pos_s, toks = None, None
                    # if the question is repeated, don't add it.
                    if a.lower() not in ans_question_set_dict:
                        ans_question_set_dict[a.lower()] = set([q.lower()])
                        filtered_qa_dict.append({'q': q, 'a': a, 'ns': ns, 'uns': uns})
                        keep_adding = True
                    elif q.lower() not in ans_question_set_dict[a.lower()]:
                        ans_question_set_dict[a.lower()].add(q.lower())
                        filtered_qa_dict.append({'q': q, 'a': a, 'ns': ns, 'uns': uns})                                                                           
                        keep_adding = True
        return filtered_qa_dict
    
    def run_qa_eval_process_local(self, document, qas_item):
        max_source_tokens = 1024
        special_token = 50259
        bsz = 32
        
        def batch_for_scorer(source_tokens_list, num_source_token_list, target_tokens_list, num_target_token_list, bsz):
            length = len(source_tokens_list)
            s = 0
            while s < length:
                e = s + bsz
                yield source_tokens_list[s:e], num_source_token_list[s:e], \
                    target_tokens_list[s:e], num_target_token_list[s:e]
                s = e

        src_tokens = self.bart.encode(document.strip(), no_bos=True, input_is_bpe=False)
        if len(src_tokens) > max_source_tokens:
            src_tokens[max_source_tokens - 1] = src_tokens[-1]
        src_tokens = src_tokens if len(src_tokens) <= max_source_tokens else src_tokens[:max_source_tokens]
        qa_tensors = []
        for qa in qas_item:
            q_tensor = self.bart.encode(qa['q'], no_bos=True, input_is_bpe=False)
            q_tensor[-1] = special_token
            a_tensor = self.bart.encode(qa['a'], no_bos=True, input_is_bpe=False)
            qa_tensors.append(torch.cat((q_tensor, a_tensor)))
        num_src_tokens = src_tokens.numel()
        src_tokens_list = [src_tokens for _ in range(len(qa_tensors))]
        num_src_token_list = [num_src_tokens for _ in range(len(qa_tensors))]
        hypos = []
        for s_list, num_s_list, t_list, num_t_list in batch_for_scorer(src_tokens_list, num_src_token_list, qa_tensors, [x.numel() for x in qa_tensors], bsz):
            if type(s_list) is not list:
                s_list = [s_list]
            if type(num_s_list) is not list:
                num_s_list = [num_s_list]
            if type(t_list) is not list:
                t_list = [t_list]
            if type(num_t_list) is not list:
                num_t_list = [num_t_list]

            dataset = LanguagePairDataset(s_list, num_s_list, self.bart.task.source_dictionary, t_list, num_t_list, self.bart.task.target_dictionary, shuffle=False)
            sample = dataset.collater(dataset)
            sample = utils.apply_to_sample(lambda tensor: tensor.cuda(), sample)
            generator = SequenceScorer(self.bart.task.target_dictionary, compute_alignment=False)
            translations = self.bart.task.inference_step(generator,[self.bart.model], sample,)
            translations = [v for _, v in sorted(zip(sample['id'].tolist(), translations))]
            hypos += translations
        qa_id = 0
        for qa in qas_item:
            hypo = hypos[qa_id]
            qa['eval_ns'] = hypo[0]['score'].item()
            qa['eval_uns'] = sum(hypo[0]['positional_scores']).item()
            special_token_loc = (hypo[0]['tokens'] == special_token).nonzero()
            ans_scores = hypo[0]['positional_scores'][special_token_loc+1:-1]
            qa['eval_a_uns'] = sum(ans_scores).item() if ans_scores.numel() > 0 else 0.0
            qa['eval_a_ns'] = qa['eval_a_uns'] * 1.0 / ans_scores.numel() if ans_scores.numel() > 0 else 0.0
            qa['eval_pos_scores'] = hypo[0]['positional_scores'].tolist()
            qa_id += 1
        return qas_item
    
    def compute_hypos_lm_score(self, qas_item):
        metrics = ['eval_ns-ns']
        hypo_avg = {}
        for metric in metrics:
            sum_per_hypo = 0.0
            count_per_hypo = 0
            for qa in qas_item:
                metric_split = metric.split('-')
                if len(metric_split) == 1:
                    value = qa[metric]
                else:
                    value = qa[metric_split[0]] - qa[metric_split[1]]
                sum_per_hypo += value
                count_per_hypo += 1
            if count_per_hypo > 0:
                hypo_avg[metric] = sum_per_hypo / count_per_hypo
            else:
                hypo_avg[metric] = -1e5
        return hypo_avg
                

    def score(self, document, claim):
        hypotheses_batch, score_batch, unnormalized_score_batch = self._sample_wrapper([claim.strip()])
        qa_list = self.filter_qas_dataset_lm_score(hypotheses_batch[0], score_batch[0], unnormalized_score_batch[0], claim.strip())
        qas_item = self.run_qa_eval_process_local(document, qa_list)
        score = self.compute_hypos_lm_score(qas_item)['eval_ns-ns']
        return score
  
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

if __name__ == "__main__":
    eval = QUALSEval()
    result = eval.score('Bob went to Beijing.', 'Bob went to Beijing.')
    pdb.set_trace()
