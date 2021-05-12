from utils import readfile as rf
from utils import writefile as wf
from utils import demographic as dg

from tqdm import tqdm
import pathlib
import json

from pyserini.search import SimpleSearcher
from pygaggle.rerank.transformer import MonoBERT
# from pygaggle.rerank.base import Query

from transformers import BertTokenizer, BertForSequenceClassification
import torch

def main():
    query_path = '../../data/TRECPM2017/topics2017.xml'
    qrel_path = '../../data/TRECPM2017/qrels.txt'
    output_path = 'runs/TREC/'
    indexing_path = '../pyserini/indexes/TRECPM2019'

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    bm25_k = 1000
    thread = 10
    bert_k = 50

    fields ={
        'contents':1.0,
        'bt':1.0,
        'dd':1.0,
        'primary_outcome':1.0,
        'intervention_name':1.0,
        'intervention_type':1.0,
        'inclusion':1.0,
        'exclusion':1.0
    }

    searcher = SimpleSearcher(indexing_path)
    searcher.set_bm25(1.2, 0.75)

    print(searcher.doc('NCT00101010').raw())

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
    hits = searcher.batch_search(queries, qids, bm25_k, thread, None, fields)
    # wf.write_hits(hits, output_path+'pyserini_dev_bm25_1000', run_name=run_name)
    #
    # # demographic filter
    # dg.filter(query_dict, hits)
    # topklist = dg.topkrank_hit(hits, bert_k, fields)
    # wf.write_hits(topklist, output_path + 'pyserini_dev_bm25_50', run_name=run_name)

    # reranker
    # texts = topklist['1']
    # for i in range(0, 10):
    #     print(f'{i+1:2} {texts[i].metadata["docid"]:15} {texts[i].score:.5f}')

    # reranker =  MonoBERT()
    # query = Query(query_dict['1']['text'])
    # reranked = reranker.rerank(query, texts)

    # for i in range(0, 10):
    #     print(f'{i+1:2} {reranked[i].metadata["docid"]:15} {reranked[i].score:.5f}')

if __name__ == '__main__':
    main()