import json


# from pygaggle.rerank.base import Text

def filter(query_dict, hits):
    for qid in hits.keys():
        for hit in hits[qid]:
            jfile = json.loads(hit.raw)
            # print(hit.docid, query_dict[qid]['age'],jfile['min_age'],jfile['max_age'])
            min_age = int(jfile['min_age']) if 'min_age' in jfile and jfile['min_age'] != 'N/A' and len(
                jfile['min_age']) > 0 else 0
            if 'max_age' in jfile and jfile['max_age'] != 'N/A' and len(jfile['max_age']) > 0 and jfile['max_age'][
                -1] != 'M':
                max_age = int(jfile['max_age'])
            elif 'max_age' in jfile and len(jfile['max_age']) > 0 and jfile['max_age'][-1] == 'M':
                max_age = 1
            else:
                max_age = 200
            # gender
            if 'gender' in jfile:
                if jfile['gender'].lower() != "all" and jfile['gender'].lower() != "both" and jfile['gender'].lower() != query_dict[qid]['gender'].lower():
                    hit.score = 0
                    # if hit.docid in qrels[qid] and int(qrels[qid][hit.docid])>0:
                    #     print(qid, query_dict[qid]['gender'].lower(), hit.docid, jfile['gender'].lower(), qrels[qid][hit.docid])
            if int(query_dict[qid]['age']) < min_age or int(query_dict[qid]['age']) > max_age:
                hit.score = 0
                # if hit.docid in qrels[qid] and int(qrels[qid][hit.docid])>0:
                #     print(qid, query_dict[qid]['age'], hit.docid, min_age, max_age, jfile['min_age'], jfile['max_age'], qrels[qid][hit.docid])


def _json2bert(jin, fields):
    out = ''
    for k in fields.keys():
        if k in jin:
            out += jin[k] + ' '
            # out += '<{}> '.format(fields[k]) + jin[k]
    return out


# def topkrank_text(hits, k, fields):
#     res = {}
#     for qid in hits.keys():
#         res[qid] = []
#         cnt = 0
#         for hit in hits[qid]:
#             if cnt > k:
#                 break
#             if hit.score > 0.000001:
#                 res[qid].append(Text(_json2bert(json.loads(hit.raw), fields), {'docid': hit.docid}, 0))
#                 cnt += 1
#     return res

def topkrank_hit(hits, k):
    res = {}
    for qid in hits.keys():
        res[qid] = []
        cnt = 0
        for hit in hits[qid]:
            if cnt >= k:
                break
            if hit.score > 0.000001:
                res[qid].append(hit)
                cnt += 1
    return res
