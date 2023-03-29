import sys
import os
import pdb
sys.path.append('/root/autodl-tmp/SSC/DataGeneration')
metric_root = '/root/autodl-tmp/SSC/Metrics'
sys.path.append(metric_root)
for folder_name, subfolders, filenames in os.walk(metric_root):
    for folder in subfolders:
        sys.path.append(os.path.join(metric_root, folder))
from DataGeneration.benchmark import SummaCBenchmark, load_dataset
from Metrics.ClozE.cloze_eval import ClozEEval
from Metrics.CoLA.cola_eval import ColaEval
from Metrics.DAE.dae_eval import DAEEval
from Metrics.FactCC.factcc_eval import FactccEval
from Metrics.FEQA.feqa_eval import FEQAEval
from Metrics.QUALS.quals_eval import QUALSEval
from Metrics.SummaC.summacconv_eval import SummaCConvEval

def evaluate_xsum():
    benchmark = SummaCBenchmark()
    benchmark.xsum = load_dataset("xsum")["test"]
    eval = ClozEEval()

if __name__ == "__main__":
    evaluate_xsum()