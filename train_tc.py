from utils import init_model
from utils.tradquery import runIRmethod_ts
from utils import readfile as rf
from utils.training import trainer
from utils import init_split
from utils.arg import argretrieve_ts
from eval import trec_eval
import pathlib
import os
import numpy as np
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def main():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    root_path = '../../data/test_collection'
    indexing_path = ['../pyserini/indexes/test_collection']
    args = argretrieve_ts()
    pretrained = args.pretrained
    lr = float(args.lr)
    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)
    isFinetune = int(args.isFinetune)
    num_epochs = int(args.num_epochs)
    bert_k = int(args.bert_k)  # 50
    query_type = args.query_type
    top_k_boolean = args.top_k_boolean
    ir_method = args.irmethod + '_ft' if isFinetune else args.irmethod  # 'bm25'
    ir_method = ir_method + '_' + query_type
    modelname = args.model_name
    num_negative = int(args.num_negative)
    path_to_trained_model = args.path_to_pretrain
    bm25_k = 1000
    num_kfold = 2

    device, output = None, None

    query_dict = {}
    qrel_dict = rf.read_qrel(root_path + '/qrels-clinical_trials.tsv')
    if 'bs' in query_type:
        rf.read_ts_topic(query_dict, root_path + '/topics-2014_2015-summary.topics')
    if 'dd' in query_type:
        rf.read_ts_topic(query_dict, root_path + '/topics-2014_2015-description.topics')
    if 'bl' in query_type:
        rf.read_ts_boolean(query_dict, top_k_boolean, root_path + '/boolean_qid.json')


    del query_dict['201428']  # non relevance

    if output is None:
        output = os.path.join("output_ts", "{}_{}_{}".format(query_type, modelname,
                                                          "pretrained_" + pretrained if pretrained else "random"))
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    # train val test split
    # dataidx = init_split.split_ts(isFinetune, list(query_dict.keys()))

    # cross-validate train val test split
    dataidx_list = init_split.split_ts_cv(list(query_dict.keys()), num_kfold)

    for i, dataidx in enumerate(dataidx_list):
        outputcv = os.path.join(output, 'cv' + str(i))
        pathlib.Path(outputcv).mkdir(parents=True, exist_ok=True)

        # Initialization model
        device, tokenizer, model = init_model.model_init(device, pretrained, isFinetune, output, path_to_trained_model)

        # run IR method
        dataloaders = runIRmethod_ts(tokenizer, dataidx, query_dict, qrel_dict, indexing_path, outputcv, bm25_k, bert_k,
                                  ir_method, batch_size,
                                  num_workers, device, num_negative)
        trainBERT = trainer(model, pretrained, outputcv, batch_size, num_epochs, dataloaders, device, isFinetune, lr)
        trainBERT.train()
        test_out_path = os.path.join(outputcv, 'pyserini_dev_{}_{}_{}.res'.format(ir_method, str(bm25_k), 'test'))
        qrel_out_path = '/'.join(test_out_path.split('/')[:-1]) + '/test_qrels.txt'
        trainBERT.test(dataidx['test'], qrel_dict, test_out_path, qrel_out_path, bert_k,
                       modelname + '_ft' if isFinetune else modelname)
        bert_out_path = '/'.join(test_out_path.split('/')[:-1]) + '/' + (
            modelname + '_ft' if isFinetune else modelname) + '.res'
        trec_eval.eval_set(qrel_out_path, test_out_path, os.path.join(outputcv, ir_method))
        trec_eval.eval_set(qrel_out_path, bert_out_path, os.path.join(outputcv, modelname))
    dirs = ['cv{}'.format(str(i)) for i in range(num_kfold)]
    trec_eval.average_all_cv_eval(output, dirs, 'overall_cv.eval')

if __name__ == '__main__':
    main()
