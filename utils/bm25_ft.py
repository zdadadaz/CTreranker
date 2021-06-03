import sys
import os
sys.path.append('.')
sys.path.append('..')
from pyserini.search import SimpleSearcher
from utils import readfile as rf
import writefile as wf
import pathlib
from eval import trec_eval

class Bm25_ft():
    def __init__(self, qids_dict, qrel_out_path, query_dict, indexing_path, runname, output_path, bm25_k=1000):
        self.coef = [[0.5, 1.2, 2.0], [0.3, 0.75]]
        self.rm = [True, False]
        self.bm25_k = bm25_k
        self.run_name = runname
        self.thread = 10
        self.searcher = SimpleSearcher(indexing_path[0])
        self.qids_dict = qids_dict
        self.query_dict = query_dict
        self.output_path = output_path
        self.qrel_out_path = qrel_out_path

    def search_ft(self):
        queries = []
        qids = []
        for idx in self.qids_dict:
            queries.append(self.query_dict[str(idx)]['text'])
            qids.append(str(idx))

        for rm in self.rm:
            if rm:
                self.searcher.set_rm3()
            else:
                if self.searcher.is_using_rm3():
                    self.searcher.unset_rm3()

            for k in self.coef[0]:
                for b in self.coef[1]:
                    suffix = f'{k}_{b}_{rm}'
                    self.searcher.set_bm25(k, b)
                    hits = self.searcher.batch_search(queries, qids, self.bm25_k + 500, self.thread)
                    out_path = os.path.join(self.output_path + '/runs',
                                            'pyserini_{}_{}_{}'.format(self.run_name, str(self.bm25_k), suffix))
                    run_name = '_'.join([self.run_name, str(k), str(b), str(rm)])
                    wf.write_hits(hits, out_path, self.bm25_k, run_name=run_name)
                    trec_eval.eval_set(self.qrel_out_path, out_path+'.res', os.path.join(self.output_path + '/eval', run_name))


def main():
    root_path = '../../data/test_collection'
    indexing_path = ['../pyserini/indexes/test_collection_sym']
    output_path = 'output_tc_bm25_ft_sym'
    query_types = ['bs', 'dd', 'bl', 'bs_dd', 'bs_bl', 'dd_bl', 'bs_dd_bl']
    top_k_boolean = 100
    pathlib.Path(output_path + '/runs').mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path + '/eval').mkdir(parents=True, exist_ok=True)
    qrel_path = root_path + '/qrels-clinical_trials.tsv'
    qrel_dict = rf.read_qrel(qrel_path)
    for query_type in query_types:
        query_dict = {}
        if 'bs' in query_type:
            rf.read_ts_topic(query_dict, root_path + '/topics-2014_2015-summary.topics')
        if 'dd' in query_type:
            rf.read_ts_topic(query_dict, root_path + '/topics-2014_2015-description.topics')
        if 'bl' in query_type:
            rf.read_ts_boolean(query_dict, top_k_boolean, root_path + '/boolean_qid.json')
        bm25ft = Bm25_ft(qrel_dict, qrel_path, query_dict, indexing_path, query_type, output_path)
        bm25ft.search_ft()


if __name__ == '__main__':
    main()