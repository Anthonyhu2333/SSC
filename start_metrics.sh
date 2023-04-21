cd /root/autodl-tmp/SSC/Metrics/ClozE; python flask_cloze.py &
cd /root/autodl-tmp/SSC/Metrics/DAE/checkpoint/stanford-corenlp-full-2018-02-27; java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer &
cd /root/autodl-tmp/SSC/Metrics/DAE; python flask_dae.py &
cd /root/autodl-tmp/SSC/Metrics/FactCC; python flask_factcc.py &
cd /root/autodl-tmp/SSC/Metrics/FEQA; CUDA_VISIBLE_DEVICES=1 python flask_feqa.py &
cd /root/autodl-tmp/SSC/Metrics/QUALS; python flask_quals.py &
cd /root/autodl-tmp/SSC/Metrics/DAE_doc; CUDA_VISIBLE_DEVICES=1 python flask_dae.py &
cd /root/autodl-tmp/SSC/Metrics/SummaC; python flask_summac.py &