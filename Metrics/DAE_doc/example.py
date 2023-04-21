# 本脚本为DAEEvaluator使用样例，请勿修改。

from dae_eval import DAEEvaluator



# 启动前，请确保已经运行如下脚本
# cd stanford-corenlp-full-2018-02-27
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

scorer = DAEEvaluator()


documents = ["discovered on land at north elmham , near dereham , the circa 600 ad coin was created by french rulers of the time to increase their available currency . adrian marsden , finds officer based at norwich castle museum , said the object was probably buried with its owner . the pendant was declared treasure by the norfolk coroner on wednesday . mr marsden added : ` ` this is an early copy of a byzantine gold coin made in france . ` ` the merovingians - lsb - french rulers - rsb - created copies of byzantine coins from their bullion as there was n ' t enough coinage coming in from the eastern roman empire . how many of these copies were ` official ' currency is hard to say . ' ' the 23 . 5 mm diameter pendant , created from an imitation of a gold solidus of emperor maurice tiberius - lrb - 582 - 602 ad - rrb - , features a suspension loop with three longitudinal ribs having been soldered to the edge of the coin immediately above the emperor ' s head . ` ` what ' s interesting is you have somebody in france copying a byzantine coin which then also followed the trend of turning it into jewellery . ' ' mr marsden said the coin was likely to have come to england as a result of export trade at the time . ` ` we see very few of these so it ' s an interesting find and one that we will hope to acquire for the norwich castle museum collection . ' ' other items declared treasure at the coroner ' s inquest include an early - medieval carolingian - style silver mount found in barnham broom , a hoard of 150 roman coins discovered in quidenham and an early - medieval biconical gold bead which would have been worn on high - status necklaces ."]
summaries = ['a rare roman gold pendant , believed to be one of the oldest in the world , has been found in norfolk .']

results = scorer.score(documents, summaries)

print(results)