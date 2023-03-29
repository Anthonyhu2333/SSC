import sys
sys.path.append('..')
from metric_utils import Base_Eval
from fairseq.models.bart import BARTModel
import torch
import pdb

class QUALSEcal(Base_Eval):
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


    def _sample_wrapper(self, sentences, beam=60, verbose=False, return_all=False,
               input_is_bpe=False, return_token_scores=False, **kwargs):
        model = self.bart
        hypotheses_batch, score_batch, unnormalized_score_batch= model.sample(
            sentences=sentences,
            beam=beam,
            verbose=verbose,
            return_all=return_all,
            input_is_bpe=input_is_bpe,
            return_token_scores=return_token_scores,
            **kwargs
        )
        return hypotheses_batch, score_batch, unnormalized_score_batch

    def score(self, document, claim):
        hypotheses_batch, score_batch, unnormalized_score_batch = self._sample_wrapper(
            sentences=[claim.strip()],
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
        print(hypotheses_batch)
        #filter
    
    def evaluate_file(self, file_path):
        pass
    
    def check_key_word(self, document, claim, keyword):
        return None
    
    def check_key_words(self, document, claim):
        return None

if __name__ == "__main__":
    eval = QUALSEcal()
    eval.score('Bob went to Beijing.', 'Bob went to Beijing.')
