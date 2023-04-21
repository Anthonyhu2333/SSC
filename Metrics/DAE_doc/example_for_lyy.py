

from dae_eval import DAEEvaluator
import json


# 启动前，请确保已经运行如下脚本
# cd stanford-corenlp-full-2018-02-27
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

scorer = DAEEvaluator()

data = []
with open(f'/mnt/DataDrive/liyiyang/Correction-research/data_filter/models/BART/generated/xsum.large.generate.merged.full.jsonl', 'r', encoding='utf8') as file:
    for line in file:
        data.append(json.loads(line.strip()))

documents = []
summaries = []

for document, summary in zip([d['document'] for d in data], [d['hypothesis'] for d in data]):
    # for sentence in summary:
    documents.append(document)
    summaries.append(summary)


results = scorer.score(documents, summaries)

print(sum(results)/len(results))