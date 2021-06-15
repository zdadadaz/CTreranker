from utils import readfile as rf


def write_queries(queris, out_path, type='tsv'):
    if type == 'csv':
        qrel_out = ['#nct_id,eligibility_type,criterion\n']
    else:
        qrel_out = ['#nct_id\teligibility_type\tcriterion\n']
    for qid in queris.keys():
        if type=='csv':
            text = "{},\"{}\"\n".format(str(qid), queris[qid]["text"].strip().replace('\"', '\''))
        elif type=='tsv':
            text = "{}\t{}\t{}\n".format(str(qid),'inclusion', queris[qid]["text"].strip().replace('\"', '\''))
        qrel_out.append(text)

    with open(out_path + '.' + type, "w") as f:
        f.writelines(qrel_out)

def write_cond(conds, out_path, type='tsv'):
    if type == 'csv':
        qrel_out = ['#nct_id,eligibility_type,criterion\n']
    else:
        qrel_out = ['#nct_id\teligibility_type\tcriterion\n']
    for idx, cond in enumerate(conds):
        if type=='csv':
            text = "{},\"{}\"\n".format(str(qid), cond.strip().replace('\"', '\''))
        elif type=='tsv':
            text = "{}\t{}\t{}\n".format(str(qid),'inclusion', queris[qid]["text"].strip().replace('\"', '\''))
        qrel_out.append(text)

    with open(out_path + '.' + type, "w") as f:
        f.writelines(qrel_out)

def query2csv():
    query_type = ['dd']
    root_path = '../../data/test_collection'
    # out_path = '../../data/test_collection/topic_summary.csv'
    out_path = '../../data/test_collection/topic_discription'
    type = 'tsv'
    query_dict = {}
    if 'bs' in query_type:
        rf.read_ts_topic(query_dict, root_path + '/topics-2014_2015-summary.topics')
    if 'dd' in query_type:
        rf.read_ts_topic(query_dict, root_path + '/topics-2014_2015-description.topics')
    write_queries(query_dict, out_path, type)
