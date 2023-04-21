
import json

from tqdm import tqdm
from dae_eval import DAEEvaluator
from scipy.stats import pearsonr

def save_to_jsonl(data, path):
    with open(path, 'w', encoding='utf8') as file:
        for d in data:
            file.write(json.dumps(d)+'\n')

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

scorer = DAEEvaluator()


test = load_jsonl('/mnt/DataDrive/liyiyang/Correction-research/data_filter/data/processed/xsum/test.jsonl')

processed_test = []
for sample in tqdm(test, desc='Filtering with DAE', ncols=150):
    scores = []
    for sentence in sample['summary']:
        score = scorer.score_one_sample(sample['document'], sentence)
        score = -1 if score is None else score
        scores.append(score)

    processed_test.append({
        'document': sample['document'],
        'summary': sample['summary'],
        'scores': scores
    })

save_to_jsonl(processed_test, '/mnt/DataDrive/liyiyang/Correction-research/data_filter/filtered/filtered_with_dae/xsum/test.jsonl')



train = load_jsonl('/mnt/DataDrive/liyiyang/Correction-research/data_filter/data/processed/xsum/train.jsonl')

processed_train = []
for sample in tqdm(train, desc='Filtering with DAE', ncols=150):
    scores = []
    for sentence in sample['summary']:
        
        score = scorer.score_one_sample(sample['document'], sentence)
        score = -1 if score is None else score
        scores.append(score)

    processed_train.append({
        'document': sample['document'],
        'summary': sample['summary'],
        'scores': scores
    })

save_to_jsonl(processed_train, f'/mnt/DataDrive/liyiyang/Correction-research/data_filter/filtered/filtered_with_dae/xsum/train.jsonl')