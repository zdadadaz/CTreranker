from utils import init_model
from utils.tradquery import runIRmethod
from utils import readfile as rf
from utils.training import trainer
from utils import init_split
from utils.arg import argretrieve
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
    # modelname = 'bm25_BERT', pretrained = 'base', lr_step_period = None, output = None
    root_path = '../../data/TRECPM2017'
    # indexing_path = ['../pyserini/indexes/TRECPM2017_txt', '../pyserini/indexes/TRECPM2019_txt']
    indexing_path = ['../pyserini/indexes/TRECPM2017_txt_v2', '../pyserini/indexes/TRECPM2019_txt_v2']
    args = argretrieve()
    pretrained = args.pretrained
    lr = float(args.lr)
    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)
    isFinetune = int(args.isFinetune)
    num_epochs = int(args.num_epochs)
    bert_k = int(args.bert_k)  # 50
    eval_year = args.year  # '2019'
    ir_method = args.irmethod + '_ft' if isFinetune else args.irmethod  # 'bm25'
    modelname = args.model_name
    num_negative = int(args.num_negative)
    path_to_trained_model = args.path_to_pretrain
    bm25_k = 1000

    device, output = None, None

    assert eval_year == '2017' or eval_year == '2018' or eval_year == '2019'

    # need combine 2017-2019 as a whole query
    topics_nums_by_year = [29, 50, 38]
    topics = []
    qrels = []
    for i in ['2017', '2018', '2019']:
        qrels.append(rf.read_qrel(root_path + '/qrels_{}.txt'.format(i)))
        topics.append(rf.read_topics(root_path + '/topics_{}.xml'.format(i)))
    # delete topic without relevant doc
    del topics[0]['10']  # 2017
    del topics[2]['32']  # 2019
    del topics[2]['33']  # 2019
    query_dict = rf.concat_topics(topics)
    qrel_dict = rf.concat_topics(qrels)

    if output is None:
        output = os.path.join("output", "{}_{}_{}".format(eval_year, modelname,
                                                          "pretrained_" + pretrained if pretrained else "random"))
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    # Initialization model
    device, tokenizer, model = init_model.model_init(device, pretrained, isFinetune, output, path_to_trained_model)

    # train val test split
    dataidx = init_split.split(eval_year, isFinetune)

    # # run IR method
    dataloaders = runIRmethod(tokenizer, dataidx, query_dict, qrel_dict, indexing_path, output, bm25_k, bert_k,
                              ir_method, batch_size,
                              num_workers, device, num_negative)
    trainBERT = trainer(model, pretrained, output, batch_size, num_epochs, dataloaders, device, isFinetune, lr)
    trainBERT.train()
    test_out_path = os.path.join(output, 'pyserini_dev_demofilter_{}_{}_{}.res'.format(ir_method, str(bm25_k), 'test'))
    qrel_out_path = '/'.join(test_out_path.split('/')[:-1]) + '/test_qrels.txt'
    trainBERT.test(dataidx['test'], qrel_dict, test_out_path, qrel_out_path, bert_k,
                   modelname + '_ft' if isFinetune else modelname)
    bert_out_path = '/'.join(test_out_path.split('/')[:-1]) + '/' + (
        modelname + '_ft' if isFinetune else modelname) + '.res'
    trec_eval.eval_set(qrel_out_path, test_out_path, os.path.join(output, ir_method))
    trec_eval.eval_set(qrel_out_path, bert_out_path, os.path.join(output, modelname))


if __name__ == '__main__':
    main()
