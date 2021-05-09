
from eval import trec_eval

qrel_out_path= 'output/BERTbase_2019_pretrained_BERTbase/test_qrels.txt'
res = 'output/BERTbase_2019_pretrained_BERTbase/pyserini_dev_demofilter_BERTbase_1000_test.res'
trec_eval.trec_eval(qrel_out_path, res)