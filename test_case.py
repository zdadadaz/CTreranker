from test import datagen
from test.get_bm25_neg import get_judge_neg
from test.inference import run_inference_test, run_inference_neg_test
from test.lr import test_warmup

def main():
    # a = datagen.TestStringMethods()
    # a.test_train_dataloader()
    # a.test_test_dataloader()
    # a.test_train_trec_dataloader()
    # get_judge_neg()
    # run_inference_test()
    # run_inference_neg_test()
    test_warmup()

if __name__ == '__main__':
    main()