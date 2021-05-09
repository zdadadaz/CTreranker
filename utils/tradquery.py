from pyserini.search import SimpleSearcher
from utils import writefile as wf
from utils import demographic as dg
import os

class Tradquery():
    def __init__(self, query_dict, qids, indexing_path, output_path, bm25_k, bert_k, run_name, suffix):
        self.coef = [1.2, 0.75]
        self.bm25_k = bm25_k
        self.bert_k = bert_k
        self.run_name = run_name
        self.thread = 10
        self.fields = {
            'contents': 1.0,
            'bs': 1.0,
            'dd': 1.0,
            'primary_outcome': 1.0,
            'intervention_name': 1.0,
            'criteria': 1.0
        }
        self.suffix = suffix
        self.topklist = None
        self.output_path = output_path

        self.searcher = SimpleSearcher(indexing_path)
        self.queries = []
        self.qids = qids
        self.query_dict = query_dict
        self.queries = [query_dict[str(idx)]['text'] for idx in qids]
        self.search()

    def search(self):
        self.searcher.set_bm25(self.coef[0], self.coef[1])
        hits = self.searcher.batch_search(self.queries, self.qids, self.bm25_k, self.thread, None, self.fields)
        out_path = os.path.join(self.output_path, 'pyserini_dev_{}_{}_{}'.format(self.run_name, str(self.bm25_k), self.suffix))
        wf.write_hits(hits, out_path, run_name=self.run_name)

        # demographic filter
        dg.filter(self.query_dict, hits)
        out_path = os.path.join(self.output_path, 'pyserini_dev_demofilter_{}_{}_{}'.format(self.run_name, str(self.bm25_k),  self.suffix))
        wf.write_hits(hits, out_path,excludeZero = True, run_name=self.run_name)
        self.topklist = dg.topkrank_hit(hits, self.bert_k, self.fields)

        # wf.write_hits(self.topklist, os.path.join(self.output_path, 'pyserini_dev_{}_{}_{}'.format(self.run_name, self.bert_k, self.suffix)), run_name=self.run_name)

