from src import datagen
from transformers import AutoTokenizer
import torch
import os
from utils import gendata

class TestStringMethods:

    def test_train_dataloader(self):
        os.system('pwd')
        year = '2019'
        phase = 'train'
        irmethod = 'bm25'
        model_type = "bert-large-uncased"
        cache_dir = "../../cache"
        collection_path = "data/year/collection/{}_{}_{}.txt".format(year, phase, irmethod)
        # train path
        topk_results_path = "runs/{}_bm25_{}/pyserini_dev_demofilter_bm25_1000_{}.res".format(year, phase, phase)
        path_to_qrel = "data/year/qrels_{}_{}.txt".format(year, phase)
        path_to_pos_qrel = "data/year/pos_qrels_{}_{}.txt".format(year, phase)
        path_to_query = "data/year/queries_{}_{}.txt".format(year, phase)
        num_neg_per_pos = 4
        batch_size = 2
        tokenizer = AutoTokenizer.from_pretrained(model_type, cache_dir=cache_dir)
        train_set = datagen.TrecPMTrainset(topk_results_path,
                                           path_to_pos_qrel,
                                           path_to_query,
                                           collection_path,
                                           tokenizer,
                                           num_neg_per_pos=num_neg_per_pos)
        dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=1, pin_memory=(False))
        # for i, item in enumerate(dataloader):
        #     print(item)
            # break

    def test_test_dataloader(self):
        os.system('pwd')
        year = '2019'
        phase = 'test'
        irmethod = 'bm25'
        model_type = "bert-base-uncased"
        cache_dir = "../../cache"
        collection_path = "data/year/collection/{}_{}_{}.txt".format(year, phase, irmethod)
        # train path
        topk_results_path = "runs/{}_bm25_{}/pyserini_dev_demofilter_bm25_1000_{}.res".format(year, phase, phase)
        path_to_qrel = "data/year/qrels_{}_{}.txt".format(year, phase)
        path_to_pos_qrel = "data/year/pos_qrels_{}_{}.txt".format(year, phase)
        path_to_query = "data/year/queries_{}_{}.txt".format(year, phase)
        num_neg_per_pos = 4
        batch_size = 2
        tokenizer = AutoTokenizer.from_pretrained(model_type, cache_dir=cache_dir)
        train_set = datagen.TrecPMDevset(topk_results_path,
                                           path_to_qrel,
                                           path_to_query,
                                           collection_path,
                                           tokenizer)
        dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=1, pin_memory=(False))
        # for i, item in enumerate(dataloader):
        #     print(item)
        #     break

    def test_train_trec_dataloader(self):
        os.system('pwd')
        year = '2019'
        phase = 'train'
        irmethod = 'bm25'
        model_type = "bert-large-uncased"
        cache_dir = "../../cache"
        collection_path = "data/year/collection/{}_{}_{}.txt".format(year, phase, irmethod)
        # train path
        topk_results_path = "runs/{}_bm25_{}/pyserini_dev_demofilter_bm25_1000_{}.res".format(year, phase, phase)
        path_to_qrel = "data/year/qrels_{}_{}.txt".format(year, phase)
        path_to_pos_qrel = "data/year/pos_qrels_{}_{}.txt".format(year, phase)
        path_to_query = "data/year/queries_{}_{}.txt".format(year, phase)
        num_neg_per_pos = 4
        batch_size = 2
        tokenizer = AutoTokenizer.from_pretrained(model_type, cache_dir=cache_dir)
        train_set = gendata.TrecPMDataset(topk_results_path,
                                           path_to_pos_qrel,
                                           path_to_query,
                                           collection_path,
                                           tokenizer,
                                           phase=phase,
                                           num_neg_per_pos=num_neg_per_pos)
        dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=1, pin_memory=(False))
        for i, item in enumerate(dataloader):
            print(item)

