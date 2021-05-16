import torch
from utils import readfile as rf
import random
import gc
from tqdm import tqdm
import copy


class TrecPMTrainset(torch.utils.data.Dataset):
    def __init__(self,
                 path_to_top_results,
                 path_to_pos_qrel,
                 path_to_query,
                 path_to_collection,
                 tokenizer,
                 num_neg_per_pos=4,
                 num_epochs=1
                 ):
        self.qrel_pos_dict = rf.read_qrel(path_to_pos_qrel)
        self.all_top_results = rf.read_result_topK(path_to_top_results, self.qrel_pos_dict.keys())
        self.tokenizer = tokenizer
        self.queries = rf.read_queries_list(path_to_query)
        self.collection = rf.read_collection(path_to_collection)
        self.num_neg_per_pos = num_neg_per_pos

        print('path_to_collection', path_to_collection)

        self.data = []
        collection_doc_list = list(self.collection.keys())
        for qid in tqdm(self.qrel_pos_dict.keys(), desc="Creating training set"):
            positives = self.qrel_pos_dict[qid].keys()
            top_results = copy.deepcopy(self.all_top_results[qid])
            negatives = list(set(top_results).difference(set(positives)))
            for pos_docid in positives:
                for _ in range(num_epochs):  # repeat num_epoch times for each sample
                    self.data.append((qid, pos_docid, 1))  # positive sample
                    # sample negatives for each pos pair
                    if len(negatives) >= num_neg_per_pos:
                        rand_negatives = random.sample(negatives, num_neg_per_pos)
                    else:
                        rand_negatives = random.sample(collection_doc_list, num_neg_per_pos)
                    for neg_docid in rand_negatives:
                        self.data.append((qid, neg_docid, 0))
        del self.all_top_results, collection_doc_list, self.qrel_pos_dict
        gc.collect()

    def __getitem__(self, index):
        qid, docid, label = self.data[index]
        query = self.queries[qid]
        doc = self.collection[docid]
        label = torch.LongTensor([label])

        # out = self.tokenizer(query,
        #                      doc,
        #                      return_tensors='pt',
        #                      padding='max_length',
        #                      max_length=256,
        #                      truncation=True
        #                      )
        # return out, label
        return query, doc, label

    def __len__(self):
        return len(self.data)


class TrecPMDevset(torch.utils.data.Dataset):
    def __init__(self,
                 path_to_top_results,
                 path_to_qrel,
                 path_to_query,
                 path_to_collection,
                 tokenizer,
                 bert_k=50
                 ):
        self.qrel = rf.read_qrel(path_to_qrel)
        self.all_top_results = rf.read_result_topK_pair(path_to_top_results, self.qrel.keys(), bert_k)
        self.tokenizer = tokenizer
        self.queries = rf.read_queries_list(path_to_query)
        self.collection = rf.read_collection(path_to_collection)

    def __getitem__(self, index):
        # get all passage from qrel line
        qid, docid = self.all_top_results[index]
        label = 1 if int(self.qrel[qid][docid]) > 0 else 0
        label = torch.LongTensor([label])
        query = self.queries[qid]
        doc = self.collection[docid]

        # inputs = self.tokenizer(query,
        #                         doc,
        #                         return_tensors='pt',
        #                         padding='max_length',
        #                         max_length=256,
        #                         truncation=True
        #                         )
        # return inputs, label
        return query, doc, label

    def __len__(self):
        return len(self.all_top_results)
