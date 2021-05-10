import torch
from utils.demographic import _json2bert
import json

class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, idxlist, query_dict, qrel_dict, fields, irclass, phase):
        self.qids = []
        self.queries = []
        self.query_dict = query_dict
        self.qrel_dict = qrel_dict
        self.fields = fields
        self.searcher = irclass.searcher
        self.irclass = irclass
        # get data
        for idx in idxlist:
            self.qids.append(str(idx))
            self.queries.append(query_dict[str(idx)]['text'])
        if phase == 'test':
            X, Xdoc, self.y, self.QDoc = self._genXypair_4testing()
        else:
            X, Xdoc, self.y, self.QDoc = self._genXypair_4training()
        self.encodings = tokenizer(X, Xdoc, return_tensors='pt', padding=True, truncation=True, max_length=256)


    def _genXypair_4training(self):
        # take all positive relevance doc and choose highest negative from list
        resX = []
        resY = []
        resDoc = []
        resQDoc = []
        k = 50
        cnt_list = []
        tot = 0
        for qid in self.irclass.topklist.keys():
            X = self.query_dict[qid]['text']
            y = self.qrel_dict[qid]
            cnt = 0
            # append positive
            for docid in y.keys():
                if y[docid] == 1:
                    resQDoc.append(qid + docid)
                    resX.append(X)
                    resDoc.append(_json2bert(json.loads(self.searcher.doc(docid).raw()), self.fields))
                    if y and docid in y:
                        resY.append(y[docid])
                    else:
                        resY.append(0)
                    cnt += 1
            cnttmp = cnt
            tot += cnttmp
            # append negative
            for hit in self.irclass.hits[qid]:
                if cnt < 0:
                    break
                if hit.score < 0.0001 or hit.docid in y and y[hit.docid] == 1:
                    continue
                resQDoc.append(qid + hit.docid)
                resX.append(X)
                resDoc.append(_json2bert(json.loads(hit.raw), self.fields))
                if y and hit.docid in y:
                    resY.append(y[hit.docid])
                else:
                    resY.append(0)
                cnt -= 1
            cnt_list.append((cnttmp,cnt))
        print(cnt_list)
        print(tot,print(self.irclass.topklist))
        return resX, resDoc, resY, resQDoc

    def _genXypair_4testing(self):
        # take whatever in the top 50
        resX = []
        resY = []
        resDoc = []
        resQDoc = []
        # for remain order
        qid_sort_list = [int(i) for i in self.irclass.topklist.keys()]
        qid_sort_list.sort()
        for qid in qid_sort_list:
            qid = str(qid)
            X = self.query_dict[qid]['text']
            y = self.qrel_dict[qid]
            for hit in self.irclass.topklist[qid]:
                # text = '[CLS] '+ X + ' [SEP] ' + _json2bert(json.loads(hit.raw), self.fields)
                resQDoc.append(qid+hit.docid)
                resX.append(X)
                resDoc.append(_json2bert(json.loads(hit.raw), self.fields))
                if y and hit.docid in y:
                    resY.append(1 if y[hit.docid] > 0 else 0)  #binarize
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