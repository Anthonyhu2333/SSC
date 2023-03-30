import sys
import os
import pdb
# pdb.Pdb.skip = True
import json
sys.path.append('/root/autodl-tmp/SSC/DataGeneration')
metric_root = '/root/autodl-tmp/SSC/Metrics'
sys.path.append(metric_root)
for folder_name, subfolders, filenames in os.walk(metric_root):
    for folder in subfolders:
        sys.path.append(os.path.join(metric_root, folder))
sys.path.append('../..')
from DataGeneration.benchmark import SummaCBenchmark, load_dataset
from Metrics.ClozE.cloze_eval import ClozEEval
from Metrics.CoLA.cola_eval import ColaEval
from Metrics.DAE.dae_eval import DAEEval
from Metrics.FactCC.factcc_eval import FactccEval
from Metrics.FEQA.feqa_eval import FEQAEval
from Metrics.QUALS.quals_eval import QUALSEval
from Metrics.SummaC.summacconv_eval import SummaCConvEval
from tqdm import tqdm

# fact_eval = [ClozEEval, DAEEval, FactccEval, FEQAEval, QUALSEval, SummaCConvEval]
fact_eval = [QUALSEval]
acceptance_eval = [ColaEval]

def evaluate_frank_type():
    benchmark = SummaCBenchmark()
    frank_result = {}
    frank_type = ['RelE', 'EntE', 'CircE', 'CorefE', 'LinkE', 'OutE', 'GramE']
    for item in fact_eval:
        eval_metric = item()
        eval_metric.score('Bob went to Beijing.', 'Bob went to Beijing.')
        eval_name = str(type(eval_metric).__name__)
        frank_result[eval_name] = {}
        print('________start to evaluate on '+eval_name+'____________')
        for data_type in frank_type:
            score = []
            data = benchmark.load_frank_sentence_by_error(data_type)
            print('___________________________________________________')
            print('________start to evaluate '+data_type+'____________')
            print('___________________________________________________')
            # try:
            for d in tqdm(data):
                document = str(d['document'])
                claim = str(d['claim'])
                s = eval_metric.score(document, claim)
                score.append(s)
            # except Exception as e:
            #     print(e)
            #     score = None
            frank_result[eval_name][data_type] = score
        del eval_metric
    # with open('/root/autodl-tmp/SSC/data/score/frank_type_score_0329.json', 'w') as f:
    #     f.writelines(json.dumps(frank_result))

def evaluate_xsum():
    benchmark = SummaCBenchmark()
    benchmark.xsum = load_dataset("xsum")["test"]
    xsum = {}
    for item in fact_eval:
        eval_metric = item()
        eval_name = str(type(eval_metric).__name__)
        score = []
        print('________start to evaluate on '+eval_name+'____________')
        for index in tqdm(range(2000)):
            d = benchmark.xsum[index]
            s = eval_metric.score(str(d['document']), str(d['summary']))
            score.append(s)
        xsum[eval_name] = score
    with open('/root/autodl-tmp/SSC/data/score/xsum_2000_score_0329.json', 'w') as f:
        f.writelines(json.dumps(xsum))



if __name__ == "__main__":
    evaluate_frank_type()
    # evaluate_xsum()