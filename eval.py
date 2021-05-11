
from eval import trec_eval

qrel_out_path= 'output/2019_bm25_BERT_noclip_pretrained_base/test_qrels.txt'
# res = 'output/2019_bm25_BERT_noclip_pretrained_base/pyserini_dev_bm25_ft_1000_test.res'
res = 'output/2019_bm25_BERT_noclip_pretrained_base/pyserini_dev_demofilter_bm25_ft_1000_test.res'
outlog = 'output/2019_bm25_BERT_noclip_pretrained_base/bert'
# trec_eval.eval_set(qrel_out_path, res, outlog)


tmp = 'output/2017_bm25_BERT_pretrained_base'
out = trec_eval.combine_all_eval('output')
print(out)