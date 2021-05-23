from eval import trec_eval
from utils import readfile as rf
from utils import writefile as wf

qrel_out_path = 'output/2017_bm25_BERT_length256_neg4_v2_kw_pretrained_BioBERT/test_qrels.txt'
res = 'output/2017_bm25_BERT_length256_neg4_v2_kw_pretrained_BioBERT/bm25_BERT_length256_neg4_v2_kw_ft.res'
outlog = 'test'

trec_eval.eval_set(qrel_out_path, res, outlog)

# models = ['base', 'BioBERT', 'BlueBERT', 'ClinicalBERT', 'SciBERT']
# # models = ['base']
# outnames = ['length256_neg{}_v2_kw'.format(i) for i in range(4,21,2)] + ['length256_negall_v2_kw']
# dirs = ['{}_bm25_BERT_{}_pretrained_{}'.format(str(i), outname, model) for i in range(2017, 2020) for outname in outnames for model in models]
# print(dirs)
# out = trec_eval.combine_all_eval('output', dirs, 'year_neg_kw_bioBERT_test_eval')

# out = trec_eval.combine_all_eval('output', [], '')
