import copy
import os
import pathlib
import json
from multiprocessing import Pool, cpu_count
import readfile as rf

out_path = None

def read_csv(path_to_csv) -> dict:
    res = {}
    with open(path_to_csv, 'r') as f:
        contents = f.readlines()

    for line in contents:
        qid, txt = line.strip().split(",")
        if qid not in res:
            res[qid] = txt.replace('|',' ')
        else:
            res[qid] += ' ' + txt
    return res

def write_csv(res, output_path):
    trec = []
    for qid in res.keys():
        trec.append("{},{}\n".format(qid, res[qid]))

    with open(output_path, "w") as f:
        f.writelines(trec)

def make_csv4ctparser(fn_num_tuple):
    files = fn_num_tuple[0]
    num = fn_num_tuple[1]
    with open(os.path.join(out_path, f"{num}.csv"), "w") as fout:
        fout.write("")
    condition_map = set()
    with open(os.path.join(out_path, f"{num}.csv"), "a") as fout:
        fout.write("#nct_id,title,has_us_facility,conditions,eligibility_criteria\n")
        for file in files:
            with open(file) as f:
                fjson = json.load(f)
            bt = fjson['bt'] if 'bt' in fjson else 'NA'
            if bt != 'NA':
                bt = bt.replace('\"','\'')
            condition = fjson['condition'] if 'condition' in fjson else 'NA'
            if condition != 'NA':
                condition = condition.replace(',', '|')
                condition = condition.replace('\"', '\'')
                for c in condition.split('|'):
                    condition_map.add(c.strip())
            criteria = fjson['criteria'] if 'criteria' in fjson else 'NA'
            if criteria != 'NA':
                criteria = criteria.replace('\"','\'')

            # fout.write("{},{}\n".format(file.split('/')[-1].split('.')[0],condition))
            fout.write(
                "{},\"{}\",{},\"{}\",\"{}\"\n".format(fjson['id'], bt, 'false', condition, criteria))
            fout.flush()
    # return condition_map

def make_condition_set(fn_num_tuple):
    files = fn_num_tuple[0]
    num = fn_num_tuple[1]
    condition_map = set()
    for file in files:
        with open(file) as f:
            fjson = json.load(f)
        bt = fjson['bt'] if 'bt' in fjson else 'NA'
        if bt != 'NA':
            bt = bt.replace('\"','\'')
        condition = fjson['condition'] if 'condition' in fjson else 'NA'
        if condition != 'NA':
            condition = condition.replace(',', '|')
            condition = condition.replace('\"', '\'')
            for c in condition.split('|'):
                condition_map.add(c.strip())
        criteria = fjson['criteria'] if 'criteria' in fjson else 'NA'
        if criteria != 'NA':
            criteria = criteria.replace('\"','\'')
    return condition_map

def gen_condition_tsv():
    global out_path
    root = '../../data/test_collection/clinicaltrials_json_new/'
    out_path = '../../data/test_collection/'
    filelist = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name[0] != '.' and name.split('.')[-1] == 'json':  # and name == 'NCT00046436.xml':
                filelist.append(os.path.join(path, name))
    out = []
    cnt = idx = 0
    step = cpu_count() - 1
    tmp = []
    split = len(filelist) // step
    for f in range(0, step):
        if f == step - 1:
            out.append((copy.deepcopy(filelist[(f * split):]), f))
        else:
            out.append((copy.deepcopy(filelist[(f * split):((f + 1) * split)]), f))

    # make_csv4ctparser((out,-1))
    pool = Pool(processes=step)
    res = pool.map(make_condition_set, out)
    pool.close()

    qqq = set()
    for r in res:
        for q in r:
            qqq.add(q+'\n')

    qrel_out = ['#nct_id\teligibility_type\tcriterion\n']
    for idx, cond in enumerate(list(qqq)):
        text = "{}\t{}\t{}\n".format(str(idx),'inclusion', cond.strip().replace('\"', '\''))
        qrel_out.append(text)

    with open(os.path.join(out_path,'condition_list.txt'), "w") as f:
        f.writelines(qrel_out)

def gen_csv_for_clinicaltrialparser():
    global out_path
    root = '../../data/test_collection/clinicaltrials_json_new/'
    out_path = '../../data/test_collection/clinicaltrials_csv_test'
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
    filelist = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name[0] != '.' and name.split('.')[-1] == 'json':  # and name == 'NCT00046436.xml':
                filelist.append(os.path.join(path, name))
    out = []
    cnt = idx = 0
    step = cpu_count() - 1
    tmp = []
    split = len(filelist)//step
    for f in range(0, step):
        if f == step-1:
            out.append((copy.deepcopy(filelist[(f*split):]), f))
        else:
            out.append((copy.deepcopy(filelist[(f*split):((f+1)*split)]), f))

    # for f in filelist:
    #     if 'pos_doc' in f:
    #         out.append(f)
    # make_csv4ctparser((out,-1))
    pool = Pool(processes=step)
    res = pool.map(make_csv4ctparser, out)
    pool.close()

    # qrel_dict = rf.read_qrel('../../data/test_collection/qrels-clinical_trials.tsv')
    # a = read_csv('../../data/test_collection/clinicaltrials_csv_test/-1.csv')
    # out = {}
    # for qid in qrel_dict:
    #     for docid in qrel_dict[qid]:
    #         if docid in a:
    #             out[qid] = a[docid]
    # write_csv(out, '../../data/test_collection/clinicaltrials_csv_test/-2.csv')




if __name__ == '__main__':
    # gen_csv_for_clinicaltrialparser()
    gen_condition_tsv()
