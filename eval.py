
from eval import trec_eval

qrel_out_path= 'output/2019_bm25_BERT_pretrained_base/test_qrels.txt'
# res = 'output/2019_bm25_BERT_pretrained_base/pyserini_dev_demofilter_bm25_1000_test.res'
res = 'output/2019_bm25_BERT_pretrained_base/bm25_BERT_ft.res'
outlog = 'output/2019_bm25_BERT_pretrained_base'
trec_eval.eval_set(qrel_out_path, res, outlog)


