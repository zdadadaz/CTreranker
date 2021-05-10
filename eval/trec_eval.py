
import os
import subprocess
import platform

def eval_set(qrel, res, output = None):
    # qrel = 'output/2019_bm25_BERT_pretrained_base/test_qrels.txt'
    # res = 'output/2019_bm25_BERT_pretrained_base/pyserini_dev_demofilter_bm25_1000_test.res'
    cmd = '../trec_eval-9.0.7/trec_eval -m map -m P.5 -m P.10 {} {}'.format(qrel, res)
    os.system(cmd)


    cmd = ['../trec_eval-9.0.7/trec_eval','-m', 'map', '-m', 'P.5', '-m', 'P.10', '-m', 'ndcg' ,'-m','Rprec','-m','recip_rank', qrel, res]
    shell = platform.system() == "Windows"
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               shell=shell)
    stdout, stderr = process.communicate()
    if stderr:
        print(stderr.decode("utf-8"))
    return read_trec_output(stdout.decode("utf-8"), output)

def read_trec_output(out, output = None):
    res ={}
    for i in out.split('\n'):
        tmp = [ j.strip() for j in i.split('\tall\t')]
        if len(tmp) <2:
            continue
        res[tmp[0]] = tmp[1]
    if output:
        with open(os.path.join(output, "log.csv"), "a") as f:
            for key in res:
                txt = key.ljust(20)
                f.write('{}:{}\n'.format(txt, res[key]) )
            f.flush()
    return res
