from tqdm import tqdm
import xml.etree.ElementTree as ET
from collections import defaultdict

def read_qrel(path_to_qrel) -> dict:
    """
    return a dictionary that maps qid, docid pair to its relevance label.
    """
    qrel = {}
    with open(path_to_qrel, 'r') as f:
        contents = f.readlines()

    for line in contents:
        if path_to_qrel.strip().split(".")[-1] == 'txt':
            qid, _, docid, rel = line.strip().split(" ")
        elif path_to_qrel.strip().split(".")[-1] == 'tsv':
            qid, _, docid, rel = line.strip().split("\t")
        if qid in qrel.keys():
            pass
        else:
            qrel[qid] = {}
        qrel[qid][docid] = int(rel)

    return qrel

def read_topics(path_to_topics)->dict:
    '''
    return a dict that maps qid, content pair
    '''
    topics = defaultdict(dict)
    tree = ET.parse(path_to_topics)
    root = tree.getroot()
    for topic in root:
        idx = topic.attrib['number']
        for c in topic:
            if c.tag.lower() == 'demographic':
                topics[idx]['age'] = int(c.text.split('-')[0])
                topics[idx]['gender'] = c.text.split(' ')[-1]
            else:
                topics[idx][c.tag] = c.text
                if 'text' not in topics[idx]:
                    topics[idx]['text'] = c.text
                else:
                    topics[idx]['text'] += ' '+c.text
    return topics

def concat_topics(arr):
    out = {}
    cnt = 1
    yr = 2017
    # num_topic = [30,50,40]
    for j in range(len(arr)):
        # for i in range(1,num_topic[j]+1):
        #     if str(i) in arr[j].keys():
        #     else:
        #       out[str(cnt)] = None
        for i in arr[j]:
            arr[j][str(i)]['year'] = str(yr + j)
            arr[j][str(i)]['orgid'] = str(i)
            arr[j][str(i)]['id'] = str(cnt)
            out[str(cnt)] = arr[j][str(i)]
            cnt += 1
    return out


def read_result(path_to_result)->dict:
    '''
    return a dict that maps qid ranking result
    '''
    assert path_to_result.strip().split(".")[-1] == 'res'

    res = {}
    with open(path_to_result, 'r') as f:
        contents = f.readlines()

    for line in tqdm(contents, desc="Loading results"):
        qid, _, docid, rank, score, name = line.strip().split("\t")
        if qid in res.keys():
            pass
        else:
            res[qid] = {'docid':[], 'score':[], 'rank':[] }
        res[qid]['docid'].append(docid)
        res[qid]['rank'].append(rank)
        res[qid]['score'].append(score)
    return res


def read_eval(path_to_eval):
    assert path_to_eval.strip().split(".")[-1] == 'eval'

    res = {}
    with open(path_to_eval, 'r') as f:
        contents = f.readlines()

    for line in contents:
        m, value = line.strip().split("\t")
        res[m] = value
    return res