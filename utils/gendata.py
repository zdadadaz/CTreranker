import torch
from utils.demographic import _json2bert
import json
from utils import writefile as wf
from utils import readfile as rf
import random
from tqdm import tqdm
import copy
import gc


class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, idxlist, query_dict, qrel_dict, irclass, num_negative, phase):
        self.qids = []
        self.query_dict = query_dict
        self.qrel_dict = qrel_dict
        self.num_negative = num_negative
        fields = {
            # 'contents': 'bt',
            'bs': 'bs',
            'bt': 'bt'
        }
        # get data
        self.qids = [str(i) for i in idxlist]

        if phase == 'train':
            X, Xdoc, self.y, self.QDoc = self._genXypair_4training(irclass, query_dict, qrel_dict, fields)
        else:
            X, Xdoc, self.y, self.QDoc = self._genXypair_4testing(irclass, query_dict, qrel_dict, fields)

        self.encodings = tokenizer(X, Xdoc, return_tensors='pt', padding='max_length', truncation=True, max_length=256)

    def _genXypair_4training(self, irclass, query_dict, qrel_dict, fields):
        # take all positive relevance doc and choose highest negative from list
        resX = []
        resY = []
        resDoc = []
        resQDoc = []
        hits = irclass.get_hits()
        m = set()
        for qid in self.qids:
            X = query_dict[qid]['text']
            y = qrel_dict[qid]
            cnt = 0
            # append positive
            for docid in y.keys():
                if int(y[docid]) > 0 and docid[:3] == 'NCT':
                    m.add(docid)
                    resQDoc.append(qid + docid)
                    resX.append(X)
                    resDoc.append(_json2bert(json.loads(irclass.get_raw(qid, docid)), fields))
                    resY.append(1)
                    cnt += 1
            tmpcnt = cnt
            # cnt = -1000
            # append negative from top 1000
            for hit in hits[qid]:
                if -cnt > tmpcnt * self.num_negative:  # limit to 10 times of postive case
                    # if -cnt > 1000:
                    break
                if hit.docid in m:
                    continue
                resQDoc.append(qid + hit.docid)
                resX.append(X)
                resDoc.append(_json2bert(json.loads(hit.raw), fields))
                resY.append(0)
                cnt -= 1
        return resX, resDoc, resY, resQDoc

    def _genXypair_4testing(self, irclass, query_dict, qrel_dict, fields):
        # take whatever in the top 50
        resX = []
        resY = []
        resDoc = []
        resQDoc = []
        # for remain order
        qid_sort_list = [int(i) for i in self.qids]
        qid_sort_list.sort()
        topklist = irclass.get_topklist()
        for qid in qid_sort_list:
            qid = str(qid)
            X = query_dict[qid]['text']
            y = qrel_dict[qid]
            for hit in topklist[qid]:
                # text = '[CLS] '+ X + ' [SEP] ' + _json2bert(json.loads(hit.raw), self.fields)
                resQDoc.append(qid + hit.docid)
                resX.append(X)
                resDoc.append(_json2bert(json.loads(hit.raw), fields))
                if y and hit.docid in y:
                    resY.append(1 if int(y[hit.docid]) > 0 else 0)  # binarize
                else:
                    resY.append(0)
        return resX, resDoc, resY, resQDoc

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.y[idx]
        item["qdoc"] = self.QDoc[idx]
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class TrecPMDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path_to_top_results,
                 path_to_pos_qrel,
                 path_to_query,
                 path_to_collection,
                 tokenizer,
                 phase,
                 num_neg_per_pos=4,
                 bert_k=50
                 ):
        self.qrel_pos_dict = rf.read_qrel(path_to_pos_qrel)
        self.all_top_results = rf.read_result_topK(path_to_top_results, self.qrel_pos_dict.keys())
        self.top_k_results = rf.read_result_topK(path_to_top_results, self.qrel_pos_dict.keys(), bert_k)
        self.tokenizer = tokenizer
        self.queries = rf.read_queries_list(path_to_query)
        self.collection = rf.read_collection(path_to_collection)
        self.num_neg_per_pos = num_neg_per_pos

        print('path_to_collection', path_to_collection)
        if phase == 'train':
            X, Xdoc, self.y, self.QDoc = self._gen_data_train()
        else:
            X, Xdoc, self.y, self.QDoc = self._gen_data_test(bert_k)
        self.encodings = tokenizer(X, Xdoc, return_tensors='pt', padding='max_length', truncation=True, max_length=256)


    def _gen_data_train(self):
        collection_doc_list = list(self.collection.keys())
        X, Xdoc, y, QDoc = [], [], [], []
        for qid in tqdm(self.qrel_pos_dict.keys(), desc="Creating training set"):
            positives = self.qrel_pos_dict[qid].keys()
            top_results = copy.deepcopy(self.all_top_results[qid])
            negatives = list(set(top_results).difference(set(positives)))
            for pos_docid in positives:
                # positive sample
                X.append(qid)
                Xdoc.append(pos_docid)
                y.append(1)
                QDoc.append(qid + pos_docid)
                # sample negatives for each pos pair
                if len(negatives) >= self.num_neg_per_pos:
                    rand_negatives = random.sample(negatives, self.num_neg_per_pos)
                else:
                    rand_negatives = random.sample(collection_doc_list, self.num_neg_per_pos)
                for neg_docid in rand_negatives:
                    X.append(qid)
                    Xdoc.append(neg_docid)
                    y.append(0)
                    QDoc.append(qid + neg_docid)
        del self.all_top_results, collection_doc_list, self.qrel_pos_dict
        gc.collect()
        return X, Xdoc, y, QDoc

    def _gen_data_test(self, bert_k):
        X, Xdoc, y, QDoc = [], [], [], []
        for qid in tqdm(self.top_k_results.keys(), desc="Creating testing set"):
            docids = self.top_k_results[qid]
            for docid in docids:
                X.append(self.queries[qid])
                Xdoc.append(self.collection[docid])
                QDoc.append(qid + docid)
                if docid in self.qrel_pos_dict[qid] and int(self.qrel_pos_dict[qid][docid]) > 0:
                    y.append(1)
                else:
                    y.append(0)
        return X, Xdoc, y, QDoc

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.y[idx]
        item["qdoc"] = self.QDoc[idx]
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
