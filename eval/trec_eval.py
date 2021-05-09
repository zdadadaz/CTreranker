
import os
import subprocess
import platform

def trec_eval(qrel, res):
    # qrel = 'output/BERTbase_2019_pretrained_BERTbase/test_qrels.txt'
    # res = 'output/BERTbase_2019_pretrained_BERTbase/pyserini_dev_demofilter_BERTbase_1000_test.res'
    # cmd = '../trec_eval-9.0.7/trec_eval -m set {} {}'.format(qrel, res)
    # os.system(cmd)

    cmd = ['../trec_eval-9.0.7/trec_eval','-m', 'set', qrel, res]
    print(f'Running command: {cmd}')
    shell = platform.system() == "Windows"
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               shell=shell)
    stdout, stderr = process.communicate()
    if stderr:
        print(stderr.decode("utf-8"))
    read_trec_output(stderr.decode("utf-8"))

def read_trec_output(out):
    print(out.split('\n'))