from eval import trec_eval
from utils import readfile as rf
from utils import writefile as wf

# qrel_out_path = 'output/2017_bm25_BERT_length256_neg4_v2_kw_pretrained_BioBERT/test_qrels.txt'
# res = 'output/2017_bm25_BERT_length256_neg4_v2_kw_pretrained_BioBERT/bm25_BERT_length256_neg4_v2_kw_ft.res'
# outlog = 'test'
#
# trec_eval.eval_set(qrel_out_path, res, outlog)

# models = ['base', 'BioBERT', 'BlueBERT', 'ClinicalBERT', 'SciBERT']
# # models = ['base']
# outnames = ['length256_neg10_{}_{}_{}'.format(i,j,z) for i in ['all', 'judged','unjudged'] for j in ['bert', 'bm25', 'random'] for z in ['none','bm25','otjudged']]
# dirs = ['{}_bm25_BERT_{}_pretrained_{}'.format(str(i), outname, model) for i in range(2017, 2020) for outname in outnames for model in models]
# out = trec_eval.combine_all_eval('output_judge', dirs, 'year_neg_kw_neg10_BioBERT_judge_1500_eval')

# models = ['base', 'BioBERT', 'BlueBERT', 'ClinicalBERT', 'SciBERT']
# # models = ['base']
# outnames = [f'length256_neg10_{i}' for i in ['baseline', 'disease']]
# dirs = ['{}_bm25_BERT_{}_pretrained_{}'.format(str(i), outname, model) for i in range(2017, 2020) for outname in outnames for model in models]
# out = trec_eval.combine_all_eval('output', dirs, 'year_neg_kw_neg10_BioBERT_baseline_disease_eval')

out = trec_eval.combine_all_eval_one_dir('output_tc_bm25_ft_sym_cond_fields_pick', [], 'bm25_ft')
