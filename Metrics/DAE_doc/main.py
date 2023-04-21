

from tqdm import tqdm
from dae_eval import DAEEvaluator
from scipy.stats import pearsonr
import pdb

import json

def load_jsonl(path):
    samples = []
    with open(path, 'r', encoding='utf8') as file:
        for line in file:
            samples.append(json.loads(line.strip()))
    return samples


# 启动前，请确保已经运行如下脚本
# cd stanford-corenlp-full-2018-02-27
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

scorer = DAEEvaluator()

# samples = load_jsonl('/mnt/DataDrive/liyiyang/ClozE-research-version-2.0/data/benchmark/test.jsonl')
with open('/root/autodl-tmp/SSC/frank_sentence.json', 'r') as f:
    samples = json.load(f)

groups = {}

overall_annotations = []
overall_predictions = []


for sample in tqdm(samples, ncols=150):

    scores = scorer.score([sample['document']], [sample['claim']], use_tqdm=False)
    pdb.set_trace()

    kind = sample['kind']
    prediction = scores[0]
    annotation = sample['score']

    if kind not in groups:
        groups[kind] = {'annotations': [], 'predictions': []}

    groups[kind]['annotations'].append(annotation)
    groups[kind]['predictions'].append(prediction)

    overall_annotations.append(annotation)
    overall_predictions.append(prediction)


final = {}

for key in groups:
    final[key] = pearsonr(groups[key]['annotations'], groups[key]['predictions'])

final['overall'] = pearsonr(overall_annotations, overall_predictions)

print(final)
