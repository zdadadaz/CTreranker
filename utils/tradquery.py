from pyserini.search import SimpleSearcher
from utils import writefile as wf
from utils import demographic as dg
from utils.gendata import Dataset
import os
import torch
import copy

class Tradquery():
    def __init__(self, query_dict, qids, indexing_path, output_path, bm25_k, bert_k, run_name, suffix):
        self.coef = [1.2, 0.75]
        self.bm25_k = bm25_k
        self.bert_k = bert_k
        self.run_name = run_name
        self.thread = 10
        self.fields = None
        self.suffix = suffix
        self.topklist = None
        self.output_path = output_path
        if len(indexing_path) == 2: # for two indexes
            self.qid2indexes = {str(i): (0 if i < 80 else 1) for i in range(1, 118)}
            self.searcher = [SimpleSearcher(indexing_path[0]), SimpleSearcher(indexing_path[1])]
        else: # for one index
            self.qid2indexes = None
            self.searcher = [SimpleSearcher(indexing_path[0])]
        self.queries = []
        self.qids = qids
        self.query_dict = query_dict
        self.search()

    def get_raw(self, qid, docid):
        out = None
        try:
            if self.qid2indexes:
                out = self.searcher[self.qid2indexes[qid]].doc(docid).raw()
            else:  # single index
                out = self.searcher[0].doc(docid).raw()
        except:
            print('error', qid, docid)
        return out

    def get_topklist(self):
        return self.topklist

    def get_hits(self):
        return self.hits

    def get_fields(self):
        return self.fields

    def search(self):
        self.hits = {}
        for index, searcher in enumerate(self.searcher):
            searcher.set_bm25(self.coef[0], self.coef[1])
            queries = []
            qids = []
            for idx in self.qids:
                if not self.qid2indexes or (self.qid2indexes and self.qid2indexes[idx] == index):
                    queries.append(self.query_dict[str(idx)]['text'])
                    qids.append(str(idx))

            hits = searcher.batch_search(queries, qids, self.bm25_k + 500, self.thread, None)
            self.hits.update(hits)

        out_path = os.path.join(self.output_path,
                                'pyserini_dev_{}_{}_{}'.format(self.run_name, str(self.bm25_k), self.suffix))
        wf.write_hits(self.hits, out_path, self.bm25_k, run_name=self.run_name)

        # demographic filter
        dg.filter(self.query_dict, self.hits)
        out_path = os.path.join(self.output_path,
                                'pyserini_dev_demofilter_{}_{}_{}'.format(self.run_name, str(self.bm25_k), self.suffix))
        wf.write_hits(self.hits, out_path, self.bm25_k, excludeZero=True, run_name=self.run_name)
        self.topklist = dg.topkrank_hit(self.hits, self.bert_k)

class Tradquery_judged():
    def __init__(self, query_dict, qids, qrels, indexing_path, output_path, bm25_k, bert_k, run_name, suffix):
        self.coef = [1.2, 0.75]
        self.bm25_k = bm25_k
        self.bert_k = bert_k
        self.run_name = run_name
        self.thread = 10
        self.fields = None
        self.suffix = suffix
        self.topklist = None
        self.output_path = output_path
        if len(indexing_path) == 2: # for two indexes
            self.qid2indexes = {str(i): (0 if i < 80 else 1) for i in range(1, 118)}
            self.searcher = [SimpleSearcher(indexing_path[0]), SimpleSearcher(indexing_path[1])]
        else: # for one index
            self.qid2indexes = None
            self.searcher = [SimpleSearcher(indexing_path[0])]
        self.queries = []
        self.qids = qids
        self.qrels = qrels
        self.query_dict = query_dict
        self.search()

    def get_raw(self, qid, docid):
        out = None
        try:
            if self.qid2indexes:
                out = self.searcher[self.qid2indexes[qid]].doc(docid).raw()
            else: # single index
                out = self.searcher[0].doc(docid).raw()
        except:
            print('error', qid, docid)
        return out

    def get_topklist(self):
        return self.topklist

    def get_hits(self):
        return self.hits

    def get_fields(self):
        return self.fields

    def get_judge_neg_from_hit(self,qids, hits, qrels):
        res = {}
        for qid in qids:
            res[qid] = []
            y = qrels[qid]
            judged_neg = set([docid for docid in y if int(y[docid])<1])
            cnt = 0
            for hit in hits[qid]:
                if cnt >= len(judged_neg):
                    break
                if hit.docid in judged_neg:
                    res[qid].append(hit)
        return res


    def search(self):
        self.hits = {}
        for index, searcher in enumerate(self.searcher):
            searcher.set_bm25(self.coef[0], self.coef[1])
            queries = []
            qids = []
            for idx in self.qids:
                if not self.qid2indexes or (self.qid2indexes and self.qid2indexes[idx] == index):
                    queries.append(self.query_dict[str(idx)])
                    qids.append(str(idx))

            hits = searcher.batch_search(queries, qids, self.bm25_k, self.thread, None)
            hits = self.get_judge_neg_from_hit(qids, hits, self.qrels)
            self.hits.update(hits)
        out_path = os.path.join(self.output_path,
                                'pyserini_dev_{}_{}_{}'.format(self.run_name, str(self.bm25_k), self.suffix))
        wf.write_hits(self.hits, out_path, self.bm25_k, run_name=self.run_name)


class Tradquery_ts():
    def __init__(self, query_dict, qids, indexing_path, output_path, bm25_k, bert_k, run_name, suffix):
        self.coef = [1.2, 0.75]
        self.bm25_k = bm25_k
        self.bert_k = bert_k
        self.run_name = run_name
        self.thread = 10
        self.fields = None
        self.suffix = suffix
        self.topklist = None
        self.output_path = output_path
        self.qid2indexes = None
        self.searcher = [SimpleSearcher(indexing_path[0])]
        self.queries = []
        self.qids = qids
        self.query_dict = query_dict
        self.search()

    def get_raw(self, qid, docid):
        out = None
        try:
            out = self.searcher[0].doc(docid).raw()
        except:
            print('error', qid, docid)
        return out

    def get_topklist(self):
        return self.topklist

    def get_hits(self):
        return self.hits

    def get_fields(self):
        return self.fields

    def search(self):
        self.hits = {}
        for index, searcher in enumerate(self.searcher):
            searcher.set_bm25(self.coef[0], self.coef[1])
            queries = []
            qids = []
            for idx in self.qids:
                queries.append(self.query_dict[str(idx)]['text'])
                qids.append(str(idx))

            hits = searcher.batch_search(queries, qids, self.bm25_k + 500, self.thread, None)
            self.hits.update(hits)

        out_path = os.path.join(self.output_path,
                                'pyserini_dev_{}_{}_{}'.format(self.run_name, str(self.bm25_k), self.suffix))
        wf.write_hits(self.hits, out_path, self.bm25_k, run_name=self.run_name)

        # demographic filter
        # dg.filter(self.query_dict, self.hits)
        # out_path = os.path.join(self.output_path,
        #                         'pyserini_dev_demofilter_{}_{}_{}'.format(self.run_name, str(self.bm25_k), self.suffix))
        # wf.write_hits(self.hits, out_path, self.bm25_k, excludeZero=True, run_name=self.run_name)
        self.topklist = dg.topkrank_hit(self.hits, self.bert_k)


def runIRmethod(tokenizer, dataidx, query_dict, qrel_dict, indexing_path, output, bm25_k, bert_k, IR_method, batch_size,
                num_workers, device, num_negative):
    dataloaders = {}
    # BM25
    for phase in dataidx:
        if len(dataidx[phase]) == 0:
            continue
        bm25 = Tradquery(query_dict, dataidx[phase], indexing_path, output, bm25_k, bert_k, IR_method, phase)
        dataset = Dataset(tokenizer, dataidx[phase], query_dict, qrel_dict, bm25, num_negative, phase)
        if phase == 'test':
            dataloaders[phase] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                             pin_memory=(device.type == "cuda"))
        else:
            dataloaders[phase] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                             shuffle=True, pin_memory=(device.type == "cuda"),
                                                             drop_last=True)
    return dataloaders


def runIRmethod_ts(tokenizer, dataidx, query_dict, qrel_dict, indexing_path, output, bm25_k, bert_k, IR_method, batch_size,
                num_workers, device, num_negative):
    dataloaders = {}
    # BM25
    for phase in dataidx:
        if len(dataidx[phase]) == 0:
            continue
        bm25 = Tradquery_ts(query_dict, dataidx[phase], indexing_path, output, bm25_k, bert_k, IR_method, phase)
        dataset = Dataset(tokenizer, dataidx[phase], query_dict, qrel_dict, bm25, num_negative, phase)
        if phase == 'test':
            dataloaders[phase] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                             pin_memory=(device.type == "cuda"))
        else:
            dataloaders[phase] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                             shuffle=True, pin_memory=(device.type == "cuda"),
                                                             drop_last=True)
    return dataloaders
