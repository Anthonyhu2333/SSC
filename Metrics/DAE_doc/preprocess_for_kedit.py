
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


train = load_jsonl('/mnt/DataDrive/liyiyang/Correction-research/KeDit/KFold/factcloze_pipeline/T5/generated/cnndm.t5.base.corrected.kedit.jsonl')

processed_train = []
for sample in tqdm(train, desc='Filtering with DAE', ncols=150):
    scores = []
    for sentence in sample['corrected']:
        
        score = scorer.score_one_sample(sample['document'], sentence)
        score = -1 if score is None else score
        scores.append(score)

    processed_train.append({
        'document': sample['document'],
        'corrected': sample['corrected'],
        'scores': scores
    })

save_to_jsonl(processed_train, '/mnt/DataDrive/liyiyang/Correction-research/KeDit/KFold/cnndm/generated/full_with_corrected_dae_scores.jsonl')