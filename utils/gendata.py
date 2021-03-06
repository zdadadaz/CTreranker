import torch
from utils.demographic import _json2bert
import json
from utils import writefile as wf
from utils import readfile as rf
import random
from tqdm import tqdm
import copy


class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, idxlist, query_dict, qrel_dict, irclass, num_negative, phase):
        self.qids = []
        self.query_dict = query_dict
        self.qrel_dict = qrel_dict
        self.num_negative = num_negative
        fields = {
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
        pos = {}
        neg = {}
        for qid in self.qids:
            X = query_dict[qid]['text']
            y = qrel_dict[qid]
            cnt = 0
            m = set()
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
            pos[qid] = cnt
            cnt = 0
            judge=0
            unjudge =0
            # append judged and not judged negative from top 1000
            for hit in hits[qid]:
                if cnt >= tmpcnt * self.num_negative:  # limit to num_negative times of positive case
                    break
                if hit.docid in m or hit.score <= 0.000001:  # not relevant & not the removed doc.
                    continue
                if hit.docid not in y:
                    unjudge += 1
                else:
                    judge += 1
                resQDoc.append(qid + hit.docid)
                resX.append(X)
                resDoc.append(_json2bert(json.loads(hit.raw), fields))
                resY.append(0)
                cnt += 1
            negatives = list(set(list(y.keys())).difference(m))
            neg[qid] = len(negatives)
            # print('qid', qid, 'pos', pos[qid], 'judged neg', neg[qid], 'bm25_judge', judge, 'bm25_unjudge', unjudge)

        # print('tot pos', sum([pos[qid] for qid in pos]), 'tot neg', sum([neg[qid] for qid in neg]))
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


class Dataset_judge_experiment(Dataset):
    def __init__(self, tokenizer, idxlist, query_dict, qrel_dict, irclass, num_negative, phase, task,
                 path_to_top_results, path_to_all_doc, path_to_top_neg_results):
        self.qids = []
        self.query_dict = query_dict
        self.qrel_dict = qrel_dict
        self.all_judged_doc = set([docid for i in qrel_dict if i in idxlist for docid in qrel_dict[i]])
        self.num_negative = num_negative
        fields = {
            'bs': 'bs',
            'bt': 'bt'
        }
        # get data
        self.qids = [str(i) for i in idxlist]
        self.all_top_results = rf.read_result_topK(path_to_top_results)
        self.all_doc = [rf.read_doc_names(path_to_all_doc[0]), rf.read_doc_names(path_to_all_doc[1])]
        self.judged_negative_by_retrieved = rf.read_result_topK(path_to_top_neg_results)
        if phase == 'train':
                X, Xdoc, self.y, self.QDoc = self._genXypair_4training_judge_exp(irclass, query_dict, qrel_dict, fields, task)
        else:
            X, Xdoc, self.y, self.QDoc = self._genXypair_4testing(irclass, query_dict, qrel_dict, fields)

        self.encodings = tokenizer(X, Xdoc, return_tensors='pt', padding='max_length', truncation=True, max_length=256)

    def get_all_doc_by_qid(self, qid, irclass):
        return self.all_doc[irclass.qid2indexes[qid]]

    def _genXypair_4training_judge_exp(self, irclass, query_dict, qrel_dict, fields, task):
        # task : {}_{}, {judged, unjudged} {random, bm25, bert}
        resX = []
        resY = []
        resDoc = []
        resQDoc = []
        isjudge, israndom, isbm25 = task.split('_')[0], task.split('_')[1], task.split('_')[1]
        for qid in self.qids:
            X = query_dict[qid]['text']
            y = qrel_dict[qid]
            cnt = 0
            all_pos = set()
            all_neg = set()
            # append positive
            for docid in y.keys():
                if int(y[docid]) > 0 and docid[:3] == 'NCT':
                    all_pos.add(docid)
                    resQDoc.append(qid + docid)
                    resX.append(X)
                    resDoc.append(_json2bert(json.loads(irclass.get_raw(qid, docid)), fields))
                    resY.append(1)
                    cnt += 1
                elif int(y[docid]) == 0 and docid[:3] == 'NCT':
                    all_neg.add(docid)
            tmpcnt = cnt
            # cape to the same number of negatives for judge and unjudge case
            # max_neg = min(cnt * self.num_negative, len(all_neg))
            max_neg = min(1500, cnt * self.num_negative)
            cnt = 0
            neg_set = set()
            if israndom == 'random':
                # append judged random negative
                if isjudge == 'judged': # sample from judged negative
                    negatives = list(all_neg)
                elif isjudge == 'unjudged': # sample from all unjudge collection
                    negatives = list(set(self.get_all_doc_by_qid(qid, irclass)).difference(set(y.keys())))
                else: # sample from all document
                    negatives = list(set(self.get_all_doc_by_qid(qid, irclass)).difference(all_pos))
                rand_negatives = random.sample(negatives, min(len(negatives), tmpcnt * self.num_negative))
                # print(qid, len(negatives), len(rand_negatives))
                for docid in rand_negatives:
                    if cnt >= max_neg:  # limit to 10 times of postive case
                        break
                    if isjudge == 'judged':
                        if docid not in y or docid in all_pos:  # pass relevant or not judged doc
                            continue
                    elif isjudge == 'unjudged':
                        if docid in y or docid in all_pos:  # pass relevant or judged doc
                            continue
                    else: # all
                        if docid in all_pos:  # pass relevant
                            continue
                    neg_set.add(docid)
                    resQDoc.append(qid + docid)
                    resX.append(X)
                    resDoc.append(_json2bert(json.loads(irclass.get_raw(qid, docid)), fields))
                    resY.append(0)
                    cnt += 1
            else:
                if isjudge == 'judged':
                    # use the max_neg numbers of top
                    if max_neg < len(self.judged_negative_by_retrieved[qid]):
                        all_top_results = self.judged_negative_by_retrieved[qid]
                    else:
                        # use up all judged negative
                        all_top_results = list(all_neg)
                else:
                    # use all top document from bm25/bert
                    all_top_results = self.all_top_results[qid]
                # append negative from top 1000
                for docid in all_top_results:
                    if cnt >= max_neg:  # limit to 10 times of postive case
                        break
                    if isjudge == 'judged':
                        if docid not in y or docid in all_pos:  # pass relevant or not judged doc
                            continue
                    elif isjudge == 'unjudged':
                        if docid in y or docid in all_pos:  # pass relevant or judged doc
                            continue
                    else:  # all
                        if docid in all_pos:  # pass relevant
                            continue
                    neg_set.add(docid)
                    resQDoc.append(qid + docid)
                    resX.append(X)
                    resDoc.append(_json2bert(json.loads(irclass.get_raw(qid, docid)), fields))
                    resY.append(0)
                    cnt += 1
            # append unjudge bm25
            if isbm25 == 'bm25':
                if cnt < max_neg:
                    all_top_results = set(self.all_top_results[qid]).difference(set(y.keys()))
                    for docid in all_top_results:
                        if docid not in neg_set:
                            if cnt >= max_neg:  # limit to 10 times of postive case
                                break
                            neg_set.add(docid)
                            resQDoc.append(qid + docid)
                            resX.append(X)
                            resDoc.append(_json2bert(json.loads(irclass.get_raw(qid, docid)), fields))
                            resY.append(0)
                            cnt += 1
            # append other qid judged docid
            elif isbm25 == 'otjudged':
                if cnt < max_neg:
                    all_judged_from_other = self.all_judged_doc.difference(set(y.keys()))
                    all_top_results = random.sample(list(all_judged_from_other), min(len(all_judged_from_other), max_neg-cnt))
                    for docid in all_top_results:
                        if docid not in neg_set:
                            if cnt >= max_neg:  # limit to 10 times of postive case
                                break
                            neg_set.add(docid)
                            resQDoc.append(qid + docid)
                            resX.append(X)
                            resDoc.append(_json2bert(json.loads(irclass.get_raw(qid, docid)), fields))
                            resY.append(0)
                            cnt += 1
            # print('qid', qid, 'positives', tmpcnt,'negatives ', cnt)
        return resX, resDoc, resY, resQDoc


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
        m = set()
        for qid in tqdm(self.qrel_pos_dict.keys(), desc="Creating training set"):
            positives = self.qrel_pos_dict[qid].keys()
            top_results = copy.deepcopy(self.all_top_results[qid])
            negatives = list(set(top_results).difference(set(positives)))
            cnt = 0
            for pos_docid in positives:
                # positive sample
                X.append(qid)
                Xdoc.append(pos_docid)
                y.append(1)
                QDoc.append(qid + pos_docid)
                m.add(pos_docid)
                cnt += 1
                # sample negatives for each pos pair
                # if len(negatives) >= self.num_neg_per_pos:
                #     rand_negatives = random.sample(negatives, self.num_neg_per_pos)
                # else:
                #     rand_negatives = random.sample(collection_doc_list, self.num_neg_per_pos)
                # for neg_docid in rand_negatives:
            tmpcnt = cnt
            for neg_docid in top_results:
                if -cnt > tmpcnt * self.num_neg_per_pos:
                    break
                if neg_docid in m:
                    continue
                X.append(qid)
                Xdoc.append(neg_docid)
                y.append(0)
                QDoc.append(qid + neg_docid)
                cnt -= 1

        return X, Xdoc, y, QDoc

    def _gen_data_test(self):
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
