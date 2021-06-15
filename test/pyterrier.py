import pyterrier as pt
import os
import json
from utils import readfile as rf
import pandas as pd


def index():
    pt.init(home_dir='/scratch/itee/s4575321/cache/')
    root = '../../data/TRECPM2017/clinicaltrials_json_txt_v2'
    out_index_path = '../../data/TRECPM2017/trecpm2017_pyterrier_index_separate'

    def tc_generate():
        filelist = []
        for path, subdirs, files in os.walk(root):
            for name in files:
                if name[0] != '.' and name.split('.')[-1] == 'json':  # and name == 'NCT00046436.xml':
                    filelist.append(os.path.join(path, name))
        for l in filelist:
            with open(l) as json_file:
                data = json.load(json_file)
                bs = data['bs'] if 'bs' in data else ''
                dd = data['dd'] if 'dd' in data else ''
                ot = data['ot'] if 'ot' in data else ''
                kw = data['kw'] if 'kw' in data else ''
                po = data['primary_outcome'] if 'primary_outcome' in data else ''
                it = data['intervention_type'] if 'intervention_type' in data else ''
                criteria = data['criteria'] if 'criteria' in data else ''
                gender = data['gender'] if 'gender' in data else ''
                yield {'docno': data['id'], 'text': data['contents'], 'bs': bs,
                       'dd': dd, 'ot': ot, 'kw': kw, 'po': po,
                       'it': it, 'criteria': criteria, 'gender': gender,
                       'min_age': data['min_age'], 'max_age': data['max_age']}

    iter_indexer = pt.IterDictIndexer(out_index_path, threads=20)
    indexref3 = iter_indexer.index(tc_generate(),
                                   meta=['docno', 'text', 'bs', 'dd', 'ot', 'kw', 'po', 'it', 'criteria', 'gender',
                                         'min_age', 'max_age'], meta_lengths=[20, 10000] + [4096 for i in range(10)],
                                   fields=['docno', 'text', 'bs', 'dd', 'ot', 'kw', 'po', 'it', 'criteria', 'gender',
                                         'min_age', 'max_age'])
    # indexref3 = iter_indexer.index(tc_generate(), meta=['docno', 'text'], meta_lengths=[20, 4096],
    #                                fields=['bs', 'dd', 'ot', 'kw', 'po', 'it', 'criteria', 'gender', 'min_age',
    #                                        'max_age'])


def search():
    pt.init(home_dir='/scratch/itee/s4575321/cache/')
    indexref = '../../data/TRECPM2017/trecpm2017_pyterrier_index_separate'
    index = pt.IndexFactory.of(indexref)
    topicsFile = '../../data/TRECPM2017/topics_2017.xml'
    qrelsFile = '../../data/TRECPM2017/qrels_2017.txt'
    query_dict = rf.read_topics(topicsFile)
    qids, queries = [], []
    for qid in query_dict:
        qids.append(qid)
        queries.append(query_dict[qid]['text'])
    df_query = pd.DataFrame.from_dict({'qid': qids, 'query': queries})
    qrels = pt.io.read_qrels(qrelsFile)
    # BM25_br = pt.BatchRetrieve(index, wmodel="BM25", metadata=["docno", "text"], controls={"k1": 1.2, "b": 0.75}) #, properties={"termpipelines" : "Stopwords,PorterStemmer"})
    BM25_br = pt.BatchRetrieve(index, wmodel="BM25", metadata=['docno']) >> \
              pt.text.get_text(index, ["bs", "dd"]) >> \
              pt.BatchRetrieve(index, wmodel="BM25")
              # pt.text.scorer(body_attr="bs", wmodel="BM25")

    # 'bs','dd', 'ot', 'kw', 'po', 'it', 'criteria', 'gender',
    # 'min_age', 'max_age'
    res = BM25_br.transform(df_query)
    print(res.columns)
    out = pt.Utils.evaluate(res, qrels, metrics=['map', 'recip_rank', 'P.5', 'P.10', 'P.15'])
    print(out)
