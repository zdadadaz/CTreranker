import json


# from pygaggle.rerank.base import Text

def get_topic_demographic(query):
    male_list = ['male', 'man', 'men', 'boy', 'his', 'he']
    woman_list = ['female','woman', 'women', 'girl', 'her', 'she']
    gender_table = {}
    for i in male_list:
        gender_table[i] = 'male'
    for i in woman_list:
        gender_table[i] = 'female'
    year_tag = ['-year-old', 'year old', '-year old', 'year-old', 'yo']
    month_tag = ['-month-old', 'month old', '-month old', 'month-old']
    age_wording = {'baby':1,'childhood':4, 'youth':10, 'young teenage':15, 'young woman':20, 'young man':20, 'middle teenage':22,'old teenage':30,
                   'young adult': 40, 'middle adult':50, 'old adult':60, 'young old':70, 'middle old':80, 'elder':87 }
    age_tag_list = {}
    for i in year_tag:
        age_tag_list[i] = 1
    for i in month_tag:
        age_tag_list[i] = 12

    # ethinicity = ['african', 'caucasian', 'white']
    gender = age = None
    tmp_query = query.lower().replace('.','').replace(',','').split(' ')
    for tag in gender_table:
        if tag in tmp_query:
            gender = gender_table[tag]
            break
    for yr in age_tag_list:
        if yr in query:
            try:
                tmp = query.split(yr)[0].split(' ')
                for i in tmp:
                    if len(i)>0 and i.isdigit():
                        tmp_age = float(i)/age_tag_list[yr]
                        age = '1' if tmp_age < 0 else str(int(tmp_age))
                    if age:
                        break
            except:
                print(f'age parser error for query {query}')
    if not age:
        tmp_query = query.lower()
        for yr in age_wording:
            if tmp_query.find(yr)!= -1:
                age = str(age_wording[yr])
                break
    return (gender, age)

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
            if 'gender' in jfile and 'gender' in query_dict[qid]:
                if jfile['gender'].lower() != "all" and jfile['gender'].lower() != "both" and jfile['gender'].lower() != query_dict[qid]['gender'].lower():
                    hit.score = 0
                    # if hit.docid in qrels[qid] and int(qrels[qid][hit.docid])>0:
                    #     print(qid, query_dict[qid]['gender'].lower(), hit.docid, jfile['gender'].lower(), qrels[qid][hit.docid])
            if 'age' in query_dict[qid]:
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
