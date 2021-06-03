from utils import readfile as rf
from utils import init_split
from utils import writefile as wf
from utils.tradquery import Tradquery_judged
import os
import pathlib

def get_judge_neg():

    indexing_path = ['../pyserini/indexes/TRECPM2017_txt_v2', '../pyserini/indexes/TRECPM2019_txt_v2']
    params = {
        'pretrained': 'base',
        'bm25_k': 10000,
        'bert_k': 50,
        'IR_method': 'bm25'
    }
    fields = {
        'bs': 'bs',
        'bt': 'bt'
    }

    for year in [2017, 2018]: #range(2017, 2020):
        dataidx = init_split.split(str(year), 1)
        # pathlib.Path('data/year_bm25_neg/collection/').mkdir(parents=True, exist_ok=True)
        for phase in dataidx.keys():
            # if len(dataidx[phase]) == 0:
            if phase != 'train':
                continue
            output = os.path.join("runs_bm25_neg", "{}_{}_{}".format(year, params['IR_method'], phase))
            pathlib.Path(output).mkdir(parents=True, exist_ok=True)
            qrel_dict = rf.read_qrel(f'data/year/qrels_{year}_{phase}.txt')
            query_dict = rf.read_queries_list(f'data/year/queries_{year}_{phase}.txt')

            # run IR method
            Tradquery_judged(query_dict, dataidx[phase], qrel_dict, indexing_path, output, params['bm25_k'], params['bert_k'],
                             params['IR_method'], phase)