# from test import datagen
# from test.get_bm25_neg import get_judge_neg
# from test.inference import run_inference_test, run_inference_neg_test
# from test.lr import test_warmup
# from test.pyterrier import *
from utils.bm25_ft import *
# from test.demo_test import demo_test
from test.tc_topic2csv import query2csv

def main():
    # a = datagen.TestStringMethods()
    # a.test_train_dataloader()
    # a.test_test_dataloader()
    # a.test_train_trec_dataloader()
    # get_judge_neg()
    # run_inference_test()
    # run_inference_neg_test()
    # test_warmup()
    # index()
    # search()
    bm25_ft()
    # demo_test()
    # query2csv()

if __name__ == '__main__':
    main()