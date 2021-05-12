
from eval import trec_eval

qrel_out_path= 'output/2019_bm25_BERT_noclip_pretrained_base/test_qrels.txt'
# res = 'output/2019_bm25_BERT_noclip_pretrained_base/pyserini_dev_bm25_ft_1000_test.res'
res = 'output/2019_bm25_BERT_noclip_pretrained_base/pyserini_dev_demofilter_bm25_ft_1000_test.res'
outlog = 'output/2019_bm25_BERT_noclip_pretrained_base/bert'
# trec_eval.eval_set(qrel_out_path, res, outlog)

outname = 'txt_softmax'
dirs = ['{}_bm25_BERT_{}_pretrained_base'.format(str(i), outname) for i in range(2017,2020)]
out = trec_eval.combine_all_eval('output', dirs, outname)
