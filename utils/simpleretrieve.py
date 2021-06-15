import os
from pyserini.search import SimpleSearcher
from utils import readfile as rf
from utils import writefile as wf
import pathlib
from eval import trec_eval
from pyserini.index import IndexReader
from pyserini.analysis import Analyzer, get_lucene_analyzer
from utils.demographic import get_topic_demographic
from utils.demographic import filter


def read_csv(path_to_csv) -> dict:
    res = {}
    with open(path_to_csv, 'r') as f:
        contents = f.readlines()

    for line in contents:
        qid, txt = line.strip().split(",")
        if qid not in res:
            res[qid] = txt.replace('|',' ')
        else:
            res[qid] += ' ' + txt
    return res


class Bm25_ft():
    def __init__(self, qids_dict, qrel_out_path, query_dict, indexing_path, runname, output_path, fields=None,
                 bm25_k=1000):
        # self.coef = [[0.5, 1.2, 2.0], [0.3, 0.75]]
        self.coef = [[1.2], [0.75]]
        self.rm = [False]
        self.bm25_k = bm25_k
        self.run_name = runname
        self.thread = 10
        self.searcher = SimpleSearcher(indexing_path[0])
        self.qids_dict = qids_dict
        self.query_dict = query_dict
        self.output_path = output_path
        self.qrel_out_path = qrel_out_path
        self.field_names = fields
        self.fields = {'contents': 0,
                       'bt': 10,
                       'bs': 10,
                       'dd': 10,
                       'ot': 0,
                       'kw': 0,
                       'primary_outcome': 0,
                       'intervention_type': 0,
                       'intervention_name': 0,
                       'criteria': 10,
                       'gender': 0,
                       'min_age': 0,
                       'max_age': 0,
                       'inclusion': 0,
                       'exclusion': 0,
                       'wiki_symptoms': 0,
                       'wiki_summary': 10,
                       'condition': 0
                       # 'wiki_caption': 0,
                       # 'mesh': 0,
                       # 'synonyms': 0
                       }
        if fields and fields in self.fields:
            self.fields[fields] = 10.0

    def search_ft(self):
        queries = []
        qids = []
        for idx in self.qids_dict:
            if 'text' in self.query_dict[str(idx)]:
                queries.append(self.query_dict[str(idx)]['text'])
                qids.append(str(idx))
            else:
                queries.append(self.query_dict[str(idx)])
                qids.append(str(idx))
        for rm in self.rm:
            if rm:
                self.searcher.set_rm3()
            else:
                if self.searcher.is_using_rm3():
                    self.searcher.unset_rm3()
            for k in self.coef[0]:
                for b in self.coef[1]:
                    suffix = f'{self.run_name}_{k}_{b}_{rm}_{self.field_names}'
                    self.searcher.set_bm25(k, b)
                    hits = self.searcher.batch_search(queries, qids, 1000, 10, None, self.fields)
                    # filter(self.query_dict, hits)
                    out_path = os.path.join(self.output_path + '/runs',
                                            'pyserini_{}_{}_{}'.format(self.run_name, str(self.bm25_k), suffix))
                    wf.write_hits(hits, out_path, self.bm25_k, run_name=suffix)
                    trec_eval.eval_set(self.qrel_out_path, out_path + '.res',
                                       os.path.join(self.output_path + '/eval', suffix))


def simple():
    root_path = '../../data/test_collection'
    indexing_path = ['../pyserini/indexes/test_collection_sym_cond']
    output_path = 'output_tc_bm25_ft_sym_cond_fields_manual'
    # query_types = ['bs', 'dd', 'bl', 'bs_dd', 'bs_bl', 'dd_bl', 'bs_dd_bl']
    query_types = ['mb']
    # fields = ['bs', 'bt', 'dd', 'ot', 'kw', 'primary_outcome', 'intervention_name', 'criteria', 'inclusion',
    #           'exclusion', 'condition', 'wiki_symptoms', 'wiki_summary']
    fields = ['bs','condition']
    top_k_boolean = 100
    pathlib.Path(output_path + '/runs').mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path + '/eval').mkdir(parents=True, exist_ok=True)
    qrel_path = root_path + '/qrels-clinical_trials.tsv'
    qrel_dict = rf.read_qrel(qrel_path)

    for query_type in query_types:
        if 'mb':
            query_tmp = read_csv(root_path + '/qid_pos_cond.csv')
        query_dict = {}
        rf.read_ts_topic(query_dict, root_path + '/topics-2014_2015-summary.topics')
        for qid in query_dict:
            gender, age = get_topic_demographic(query_dict[qid]['text'])
            if qid in query_tmp:
                query_dict[qid]['text'] = query_tmp[qid]
                if age:
                    query_dict[qid]['age'] = age
                if gender:
                    query_dict[qid]['gender'] = gender
    #     if 'bs' in query_type:
    #         rf.read_ts_topic(query_dict, root_path + '/topics-2014_2015-summary.topics')
    #     if 'dd' in query_type:
        rf.read_ts_topic(query_dict, root_path + '/topics-2014_2015-description.topics')
    #     if 'bl' in query_type:
    #         rf.read_ts_boolean(query_dict, top_k_boolean, root_path + '/boolean_qid.json')

        for field in fields:
            bm25ft = Bm25_ft(qrel_dict, qrel_path, query_dict, indexing_path, query_type, output_path, field)
            bm25ft.search_ft()
    out = trec_eval.combine_all_eval_one_dir(output_path, [], 'bm25_ft')



if __name__ == '__main__':
    simple()
