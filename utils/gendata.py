import torch
from utils.demographic import _json2bert
import json
from transformers import BertTokenizer

class Dataset(torch.utils.data.Dataset):
    def __init__(self, idxlist, query_dict, qrel_dict, hits, fields, searcher, phase):
        self.qids = []
        self.queries = []
        self.query_dict = query_dict
        self.qrel_dict = qrel_dict
        self.hits = hits
        self.fields = fields
        self.searcher = searcher
        # get data
        for idx in idxlist:
            self.qids.append(str(idx))
            self.queries.append(query_dict[str(idx)]['text'])
        # if phase == 'test':
        X, Xdoc, self.y, self.QDoc = self._genXypair_4testing()
        # else:
        # X, Xdoc, self.y = self._genXypair_4training()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encodings = tokenizer(X, Xdoc, return_tensors='pt', padding=True, truncation=True, max_length=256)


    def _genXypair_4training(self):
        # take all positive relevance doc and choose highest negative from list
        resX = []
        resY = []
        resDoc = []
        resQDoc = []
        k = 50
        for qid in self.hits.keys():
            X = self.query_dict[qid]['text']
            y = self.qrel_dict[qid]
            cnt = 0
            for docid in y.keys():
                if cnt >=k//2:
                    break
                if y[docid] == 1:
                    resQDoc.append(qid + hit.docid)
                    resX.append(X)
                    resDoc.append(_json2bert(json.loads(self.searcher.doc(docid).raw()), self.fields))
                    if y and docid in y:
                        resY.append(y[docid])
                    else:
                        resY.append(0)
                    cnt += 1
            # if not enough 50
            for hit in self.hits[qid]:
                # text = '[CLS] '+ X + ' [SEP] ' + _json2bert(json.loads(hit.raw), self.fields)
                if cnt >=k:
                    break
                if hit.docid in y and y[hit.docid] == 1:
                    continue
                resQDoc.append(qid + hit.docid)
                resX.append(X)
                resDoc.append(_json2bert(json.loads(hit.raw), self.fields))
                if y and hit.docid in y:
                    resY.append(y[hit.docid])
                else:
                    resY.append(0)
                cnt+=1
        return resX, resDoc, resY, resQDoc

    def _genXypair_4testing(self):
        # take whatever in the top 50
        resX = []
        resY = []
        resDoc = []
        resQDoc = []
        # for remain order
        qid_sort_list = [int(i) for i in self.hits.keys()]
        qid_sort_list.sort()
        for qid in qid_sort_list:
            qid = str(qid)
            X = self.query_dict[qid]['text']
            y = self.qrel_dict[qid]
            for hit in self.hits[qid]:
                # text = '[CLS] '+ X + ' [SEP] ' + _json2bert(json.loads(hit.raw), self.fields)
                resQDoc.append(qid+hit.docid)
                resX.append(X)
                resDoc.append(_json2bert(json.loads(hit.raw), self.fields))
                if y and hit.docid in y:
                    resY.append(1 if y[hit.docid] > 0 else 0)
                else:
                    resY.append(0)
        return resX, resDoc, resY, resQDoc

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.y[idx]
        item["qdoc"] = self.QDoc[idx]
        # item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # item["labels"] = torch.Tensor(self.y[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])