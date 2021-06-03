from utils import readfile as rf
import pathlib
from eval import trec_eval
from pygaggle.rerank.transformer import MonoBERT
from pygaggle.rerank.base import Query, Text
import os
import numpy as np


# from pyserini.search import SimpleSearcher
# from utils import writefile as wf
# from utils import demographic as dg


def write_run_file(qids, scores, docids, output_path):
    num_quries = len(qids)
    for j in range(num_quries):
        qid = qids[j]
        num_docs = len(docids[j])
        trec_lines = []
        for k in range(num_docs):
            score = scores[j][k]
            docid = docids[j][k]
            trec_lines.append(
                str(qid) + " " + "Q0" + " " + str(docid) + " " + str(k + 1) + " " + str(score) + " " + "BERT" + "\n")

        with open(output_path, "a+") as f:
            f.writelines(trec_lines)


def main():
    eval_year = 2019
    ir_method = 'bm25'

    # query_path = '../../data/TRECPM2017/topics2017.xml'
    # qrel_path = '../../data/TRECPM2017/qrels.txt'
    output_path = 'runs/TRECPM/monoBERT_{}'.format(eval_year)
    # indexing_path = '../pyserini/indexes/TRECPM2019'
    out_name = 'monoBERT_{}'.format(eval_year)

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    bm25_k = 1000
    thread = 10
    bert_k = 50

    # fields ={
    #     'contents':1.0,
    #     'bt':1.0,
    #     'dd':1.0,
    #     'primary_outcome':1.0,
    #     'intervention_name':1.0,
    #     'intervention_type':1.0,
    #     'inclusion':1.0,
    #     'exclusion':1.0
    # }

    # searcher = SimpleSearcher(indexing_path)
    # searcher.set_bm25(1.2, 0.75)

    # qrel_dict = rf.read_qrel(qrel_path)
    # query_dict = rf.read_topics(query_path)
    #
    # queries = []
    # qids = []
    # for qid in query_dict.keys():
    #     qids.append(qid)
    #     queries.append(query_dict[qid]['text'])
    #
    # run_name = 'bm25'
    # hits = searcher.batch_search(queries, qids, bm25_k, thread, None, fields)
    # wf.write_hits(hits, output_path+'pyserini_dev_bm25_1000', run_name=run_name)
    #
    # # demographic filter
    # dg.filter(query_dict, hits)
    # topklist = dg.topkrank_hit(hits, bert_k, fields)
    # wf.write_hits(topklist, output_path + 'pyserini_dev_bm25_50', run_name=run_name)

    collection_path = "data/year/collection/{}_test_{}.txt".format(eval_year, ir_method)
    dev_dataset_path = "runs/{}_{}_test/pyserini_dev_demofilter_bm25_1000_test.res".format(eval_year, ir_method)
    dev_query_path = "data/year/queries_{}_test.txt".format(eval_year)
    dev_qrel_path = "data/year/qrels_{}_test.txt".format(eval_year)

    qrel = rf.read_qrel(dev_qrel_path)
    run = rf.read_result_topK(dev_dataset_path, qrel.keys())
    query_dict = rf.read_queries_list(dev_query_path)
    collection = rf.read_collection(collection_path)

    # clear out file
    output_file_path = os.path.join(output_path, out_name + '.res')
    with open(output_file_path, "w") as f:
        f.writelines('')

    reranker = MonoBERT()
    for qid in run.keys():
        query = Query(query_dict[qid])
        res = []
        texts = []
        for idx, docid in enumerate(run[qid]):
            if idx >= bert_k:
                break
            texts.append(Text(collection[docid], {'docid': docid}, 0))
        reanked_res = reranker.rerank(query, texts)
        # for i in range(0, 10):
        #     print(f'{i + 1:2} {reanked_res[i].metadata["docid"]:15} {reanked_res[i].score:.5f}')
        total_scores = [i.score for i in reanked_res]
        zipped_lists = zip(total_scores, run[qid])
        sorted_pairs = np.array(sorted(zipped_lists, reverse=True))
        run[qid][:bert_k] = sorted_pairs[:, 1]
        scores = [i for i in range(len(run[qid]), 0, -1)]
        # write run file
        write_run_file([qid], [scores], [run[qid]], output_file_path)

    trec_eval.eval_set(dev_qrel_path, output_file_path, output_file_path)

if __name__ == '__main__':
    main()
