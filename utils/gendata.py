import torch
from utils.demographic import _json2bert
import json

class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, idxlist, query_dict, qrel_dict, irclass, phase):
        self.qids = []
        self.query_dict = query_dict
        self.qrel_dict = qrel_dict
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

        self.encodings = tokenizer(X, Xdoc, return_tensors='pt', padding=True, truncation=True, max_length=256)

    def _genXypair_4training(self, irclass, query_dict, qrel_dict, fields):
        # take all positive relevance doc and choose highest negative from list
        resX = []
        resY = []
        resDoc = []
        resQDoc = []
        hits = irclass.get_hits()
        for qid in self.qids:
            X = query_dict[qid]['text']
            y = qrel_dict[qid]
            cnt = 0
            # append positive
            for docid in y.keys():
                if int(y[docid]) > 0 and docid[:3] == 'NCT':
                    resQDoc.append(qid + docid)
                    resX.append(X)
                    resDoc.append(_json2bert(json.loads(irclass.get_raw(qid, docid)), fields))
                    resY.append(1)
                    cnt += 1
            tmpcnt = cnt
            # append negative from top 1000
            for hit in hits[qid]:
                # if -cnt > tmpcnt*10: # limit to 10 times of postive case
                # if cnt < 0:
                #     break
                if (hit.docid in y and int(y[hit.docid]) > 0):
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
                resQDoc.append(qid+hit.docid)
                resX.append(X)
                resDoc.append(_json2bert(json.loads(hit.raw), fields))
                if y and hit.docid in y:
                    resY.append(1 if int(y[hit.docid]) > 0 else 0)  #binarize
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