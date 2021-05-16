from utils import readfile as rf
from utils import init_split
from utils import writefile as wf
from utils.tradquery import Tradquery
import os
import pathlib
from utils.demographic import _json2bert
import json


def write_collections(qrel_dict, irclass, fields, bm25_k, output_path):
    res = []
    hits = irclass.get_hits()
    m = set()
    for qid in hits.keys():
        y = qrel_dict[qid]
        # positive
        for docid in y:
            if int(y[docid]) > 0 and docid[:3] == 'NCT' and docid not in m:
                m.add(docid)
                text = _json2bert(json.loads(irclass.get_raw(qid, docid)), fields)
                res.append(docid + '\t' + text + '\n')
        # negative
        cnt = 0
        for hit in hits[qid]:
            # if cnt > bm25_k+:
            #     break
            if hit.docid in m:
                continue
            m.add(hit.docid)
            text = _json2bert(json.loads(irclass.get_raw(qid, hit.docid)), fields)
            res.append(hit.docid + '\t' + text + '\n')
            cnt+=1
    with open(output_path + '.txt', "w") as f:
        f.writelines(res)

def main():
    # need combine 2017-2019 as a whole query
    topics_nums_by_year = [29, 50, 38]
    root_path = '../../data/TRECPM2017'
    topics = []
    qrels = []
    for i in ['2017', '2018', '2019']:
        qrels.append(rf.read_qrel(root_path + '/qrels_{}.txt'.format(i)))
        topics.append(rf.read_topics(root_path + '/topics_{}.xml'.format(i)))
    # delete topic without relevant doc
    del topics[0]['10']  # 2017
    del topics[2]['32']  # 2019
    del topics[2]['33']  # 2019
    query_dict = rf.concat_topics(topics)
    qrel_dict = rf.concat_topics(qrels)

    indexing_path = ['../pyserini/indexes/TRECPM2017_txt', '../pyserini/indexes/TRECPM2019_txt']
    params = {
        'pretrained': 'base',
        'bm25_k': 1000,
        'bert_k': 50,
        'IR_method': 'bm25'
    }

    fields = {
        'bs': 'bs',
        'bt': 'bt'
    }

    for year in range(2017, 2020):
        dataidx = init_split.split(str(year), 1)
        pathlib.Path('data/year/collection/').mkdir(parents=True, exist_ok=True)
        for phase in dataidx.keys():
            if len(dataidx[phase]) == 0:
                continue
            output = os.path.join("runs", "{}_{}_{}".format(year, params['IR_method'], phase))
            pathlib.Path(output).mkdir(parents=True, exist_ok=True)

            wf.write_qrels(dataidx[phase], qrel_dict, 'data/year/qrels_{}_{}.txt'.format(year, phase))
            wf.write_pos_qrels(qrel_dict, dataidx[phase], 'data/year/pos_qrels_{}_{}.txt'.format(year, phase))
            wf.write_queries(query_dict, dataidx[phase], 'data/year/queries_{}_{}.txt'.format(year, phase))

            # run IR method
            bm25 = Tradquery(query_dict, dataidx[phase], indexing_path, output, params['bm25_k'], params['bert_k'],
                             params['IR_method'], phase)
            write_collections(qrel_dict, bm25, fields, params['bm25_k'], 'data/year/collection/{}_{}_{}'.format(year, phase, params['IR_method']))


if __name__ == '__main__':
    main()
