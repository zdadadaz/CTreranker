from eval import trec_eval
from utils import readfile as rf
from utils import writefile as wf

qrel_out_path = 'output/2019_bm25_BERT_softmax_pretrained_base/test_qrels.txt'
res = 'output/2019_bm25_BERT_softmax_pretrained_base/rearrange.res'
outlog = 'output/2019_bm25_BERT_softmax_pretrained_base/bert'

# res = rf.read_result(res)
# for qid in res.keys():
#     n = len(res[qid]['score'])
#     res[qid]['score'] = [i for i in range(n, 0, -1)]
# wf.write_res(res, 'output/2019_bm25_BERT_softmax_pretrained_base/rearrange')

# trec_eval.eval_set(qrel_out_path, res, outlog)
outnames = ['length512_neg12', 'length512_neg14','length512_neg16','length512_neg18','length512_neg20']
for outname in outnames:
    dirs = ['{}_bm25_BERT_{}_pretrained_base'.format(str(i), outname) for i in range(2017, 2020)]
    out = trec_eval.combine_all_eval('output', dirs, outname)
