import sys
sys.path.append('..')
import torch
from metric_utils import Base_Eval
from fairseq.models.bart import BARTModel
from transformers import BartTokenizer, BartForConditionalGeneration, BertForQuestionAnswering, AutoTokenizer
import spacy
from tqdm import tqdm

class FEQAEval(Base_Eval):
    def __init__(self):
        self.qg_model = BARTModel.from_pretrained('/root/autodl-tmp/SSC/Metrics/FEQA/checkpoint/qg/checkpoints', checkpoint_file='checkpoint_best.pt').cuda()
        self.qa_model = BertForQuestionAnswering.from_pretrained('/root/autodl-tmp/SSC/Metrics/FEQA/checkpoint/qa/squad1.0').cuda()
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.nlp = None
        self.parser = None
        self.qg_model.half()
        self.qg_model.eval()

        self.batch_size = 64
        self.beam_size = 10
        self.max_length = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_questions(self, summaries, entities=True, phrase_types=["NP"]):
        doc_ids = []
        qa_masks = []
        tokenized_phrases = []
        for id_, summary in enumerate(summaries):
            summary = summary.strip()
            all_masked_phrases = []
            if entities:
                all_masked_phrases.extend([X.text for X in self.nlp(summary).ents])
            all_masked_phrases.extend(self._get_masked_phrases(summary,phrase_types))
            all_masked_phrases = list(set(all_masked_phrases))

            for i, masked_phrase in enumerate(all_masked_phrases):
                tokenized_summary = " ".join(nltk.word_tokenize(summary.lower()))
                tokenized_phrase = " ".join(nltk.word_tokenize(masked_phrase.lower()))

                qa_masks.append(tokenized_summary + " [SEP] " + tokenized_phrase)
                doc_ids.append(str(id_))
                tokenized_phrases.append(tokenized_phrase)
        questions = []
        for i in tqdm(range(0, len(qa_masks), self.batch_size)):
            batch = qa_masks[i:i + self.batch_size]
            hypotheses = self.qg_model.sample(batch, beam=self.beam_size, lenpen=1.0, max_len_b=self.max_length, min_len=1, no_repeat_ngram_size=3)
            questions.extend(hypotheses)
        return doc_ids, questions, tokenized_phrases

    def generate_question_by_key_word(self, summary, keyword):
        tokenized_summary = " ".join(nltk.word_tokenize(summary.lower()))
        tokenized_phrase = " ".join(nltk.word_tokenize(keyword.lower()))
        qa_masks = tokenized_summary + " [SEP] " + tokenized_phrase
        question = self.qg_model.sample([qa_masks], beam=10, lenpen=1.0, max_len_b=100, min_len=1, no_repeat_ngram_size=3)
        return question, tokenized_phrase

    def get_answers(self, summaries, documents, keyword):
        doc_ids, questions, tokenized_phrases = self.generate_questions(summary, keyword)
        answers = []
        for i in range(0, len(doc_ids), self.batch_size):
            questions_batch = questions[i:i + self.batch_size]
            doc_ids_batch = doc_ids[i:i + self.batch_size]
            documents_batch = [documents[i] for i in doc_ids_batch]
            input_text = [question+' [SEP] '+document for question, docuemnt in zip(questions_batch, documents_batch)]
            input_ids = self.tokenizer.batch_encode_plus(input_text, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
            result = self.qa_model(**input_ids)
            #如何根据result的到最后的answer
            pdb.set_trace()
            answers_start = torch.argmax(result['start_logits'])
            answers_end = torch.argmax(result['end_logits'])


    
    def get_answer_by_key_word(self, summary, document, keyword):
        question, tokenized_phrase = self.generate_questions(summary, keyword)
        input_ids = self.tokenizer.encode(question[0], document, add_special_tokens=True)[:512]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        sep_index = input_ids.index(self.tokenizer.sep_token_id)
        num_seg_a = sep_index + 1
        num_seg_b = len(input_ids) - num_seg_a
        segment_ids = [0] * num_seg_a + [1] * num_seg_b
        result = self.qa_model(torch.tensor([input_ids]).cuda(),  token_type_ids=torch.tensor([segment_ids]).cuda())
        answer_start = torch.argmax(result['start_logits'])
        answer_end = torch.argmax(result['end_logits'])
        answer = ' '.join(tokens[answer_start:answer_end + 1])
        return answer
    
    def _get_masked_phrases(self, output_summary, phrase_types=["NP"]):
        masked_phrases = []
        parse_tree = self.parser.parse(output_summary)
        for subtree in parse_tree.subtrees():
            phrases_list = [(subtree_.leaves(), subtree_.label()) for subtree_ in subtree if type(subtree_) == Tree and subtree_.label() in phrase_types]
            for phrase_tuple in phrases_list:
                phrase = phrase_tuple[0]
                phrase_type = phrase_tuple[1]
                phrase_text = " ".join(phrase)
                if len(phrase) > 0 and phrase_text not in self.stop_words:
                    masked_phrases.append(phrase_text)     
        return masked_phrases 
    
    def _compute_f1(self, a_gold, a_pred):
        gold_toks = nltk.word_tokenize(a_gold)
        pred_toks = nltk.word_tokenize(a_pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def _compute_f1_list(self, a_gold_list, a_pred_list):
        f1_list=[]
        for a_gold,a_pred in zip(a_gold_list, a_pred_list):
            f1_list.append(self._compute_f1(a_gold,a_pred))
        return np.mean(f1_list)

    def cal_score(self, claim, document):
        entity_list = [item.text for item in self.spacy(claim).ents]
        answer_list = [self.get_answer(claim, document, entity) for entity in entity_list]
        try:
            if len(entity_list) != 0:
                entity_list = [item.lower() for item in entity_list]
                f1 = self._compute_f1_list(entity_list, answer_list)
                score = f1
            else:
                score = 0
        except Exception as e:
            score = 0
        if score == np.nan:
            score = 0
        return score


    
    def score(self, docuemnt, claim):
        pass
    
    def evaluate_file(self, file_path):
        if self.nlp is None:
            self.nlp = spacy.load('en_core_web_sm')
        if self.parser is None:
            self.parser = benepar.Parser("benepar_en2")
        pass
    
    def check_key_word(self, document, claim, key_word):
        pass
    
    def check_key_words(self, document, claim, key_word):
        pass