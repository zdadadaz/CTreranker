import os
import subprocess
import platform

import pandas as pd

from utils import writefile as wf
from utils import readfile as rf
import glob
import pandas


def eval_set(qrel, res, output=None):
    # qrel = 'output/2019_bm25_BERT_pretrained_base/test_qrels.txt'
    # res = 'output/2019_bm25_BERT_pretrained_base/pyserini_dev_demofilter_bm25_1000_test.res'
    # print("Evaluation for ", output)
    # cmd = '../trec_eval-9.0.7/trec_eval -m map -m P.10 -m Rprec -m ndcg -m recip_rank {} {}'.format(qrel, res)
    cmd = '../trec_eval-9.0.7/trec_eval -q {} {} > {}'.format(qrel, res, output + '.qeval')
    os.system(cmd)

    # '-m', 'P.10' '-m', 'recall'
    cmds = [['../trec_eval-9.0.7/trec_eval', '-m', 'map', '-m', 'P.5', '-m', 'ndcg', '-m', 'Rprec', '-m', 'recip_rank',
             qrel, res],
            ['../trec_eval-9.0.7/trec_eval', '-m', 'P.10', qrel, res],
            ['../trec_eval-9.0.7/trec_eval', '-m', 'P.15', qrel, res]]
    shell = platform.system() == "Windows"
    out = ''
    for cmd in cmds:
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=shell)
        stdout, stderr = process.communicate()
        if stderr:
            print(stderr.decode("utf-8"))
        out += stdout.decode("utf-8")
    write_trec_output(out, output)


def write_trec_output(out, output=None):
    res = {}
    for i in out.split('\n'):
        tmp = [j.strip() for j in i.split('\tall\t')]
        if len(tmp) < 2:
            continue
        res[tmp[0]] = tmp[1]
    if output:
        wf.write_eval(res, output)
        run_name = output.split('/')[-1]
        output = '/'.join(output.split('/')[:-1])
        with open(os.path.join(output, "log.csv"), "a") as f:
            f.write(run_name + '\n')
            for key in res:
                txt = key.ljust(20)
                f.write('{}:{}\n'.format(txt, res[key]))
            f.flush()


def get_eval_result(output):
    res = {}
    filenames = glob.glob(os.path.join(output, '*.eval'))
    for f in filenames:
        name = f.split('/')[-1].split('.')[0]
        res[name] = rf.read_eval(f)
    return res


def combine_all_eval(output, include_dirs, outname):
    dirs = [d for r, d, f in os.walk(output) if len(d) > 0][0]
    res = {'model': [], 'method': [], 'map': [], 'Rprec': [], 'recip_rank': [], 'P_5': [], 'P_10': [], 'P_15': [],
           'ndcg': []}
    # res = {'model': [], 'method': [], 'map': [], 'Rprec': [], 'recip_rank': [], 'P_5': [], 'P_10': [], 'P_15': [],
    #        'ndcg': [], 'recall_5': [], 'recall_10': [], 'recall_15': [], 'recall_20': [], 'recall_30': [],
    #        'recall_100': [],
    #        'recall_200': [], 'recall_500': [], 'recall_1000': []}
    for d in dirs:
        if d not in include_dirs:
            continue
        tmp = get_eval_result(os.path.join(output, d))
        for i in tmp.keys():
            res['model'].append(d)
            res['method'].append(i)
            for key, value in tmp[i].items():
                if key in res:
                    res[key].append(value)
    df = pd.DataFrame.from_dict(res)
    df.to_csv(os.path.join(output, outname + '_eval.csv'), index=False)

def combine_all_eval_one_dir(output, include_files, outname):
    files = [os.path.join(r, name) for r, d, f in os.walk(output) for name in f if name[0]!= '.' and name[-5:]=='.eval']
    res = {'model': [], 'map': [], 'Rprec': [], 'recip_rank': [], 'P_5': [], 'P_10': [], 'P_15': [],
           'ndcg': [], 'recall_5': [], 'recall_10': [], 'recall_15': [], 'recall_20': [], 'recall_30': [], 'recall_100': [],
           'recall_200': [], 'recall_500': [], 'recall_1000': []}
    for f in files:
        # if f not in include_files:
        #     continue
        fn = f.split('/')[-1][:-5]
        tmp = rf.read_eval(f)
        res['model'].append(fn)
        for key, value in tmp.items():
            res[key].append(value)
    df = pd.DataFrame.from_dict(res)
    df.to_csv(os.path.join(output, outname + '_eval.csv'), index=False)

def average_all_cv_eval(output, include_dirs, outname):
    dirs = [d for r, d, f in os.walk(output) if len(d) > 0][0]
    res = {'model': [], 'method': [], 'map': [], 'Rprec': [], 'recip_rank': [], 'P_5': [], 'P_10': [], 'P_15': [],
           'ndcg': []}
    for d in dirs:
        if d not in include_dirs:
            continue
        tmp = get_eval_result(os.path.join(output, d))
        for i in tmp.keys():
            res['model'].append(d)
            res['method'].append(i)
            for key, value in tmp[i].items():
                res[key].append(float(value))
    df = pd.DataFrame.from_dict(res)

    # average
    df_out = pd.DataFrame([], [], columns=df.drop('model', axis=1).columns)
    for name in df['method'].unique():
        tmp = df[df['method'] == name].drop(['model', 'method'], axis=1).agg("mean", axis="rows").to_frame().T
        tmp['method'] = name
        df_out = df_out.append(tmp, ignore_index=True)
    df_out.to_csv(os.path.join(output, outname + '.csv'), index=False, float_format='%.4f')
