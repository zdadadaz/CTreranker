from utils import init_model
from utils.training import trainer
from utils.arg import argretrieve
from utils.gendata import TrecPMDataset
from src.inference import inference
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

    args = argretrieve()
    pretrained = args.pretrained
    lr = float(args.lr)
    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)
    isFinetune = int(args.isFinetune)
    num_epochs = int(args.num_epochs)
    bert_k = int(args.bert_k)  # 50
    eval_year = args.year  # '2019'
    ir_method = args.irmethod   # 'bm25'
    modelname = args.model_name
    num_negative = int(args.num_negative)
    path_to_trained_model = args.path_to_pretrain
    bm25_k = 1000

    # train path
    collection_path = "data/year/collection/{}_train_{}.txt".format(eval_year, ir_method)
    dataset_path = "runs/{}_{}_train/pyserini_dev_demofilter_bm25_1000_train.res".format(eval_year, ir_method)
    path_to_pos_qrel = "data/year/pos_qrels_{}_train.txt".format(eval_year)
    path_to_query = "data/year/queries_{}_train.txt".format(eval_year)
    # dev path
    dev_collection_path = "data/year/collection/{}_test_{}.txt".format(eval_year, ir_method)
    dev_dataset_path = "runs/{}_{}_test/pyserini_dev_demofilter_bm25_1000_test.res".format(eval_year, ir_method)
    dev_query_path = "data/year/queries_{}_test.txt".format(eval_year)
    dev_qrel_path = "data/year/qrels_{}_test.txt".format(eval_year)
    qrel_out_path = 'data/year/qrels_{}_test.txt'.format(eval_year)

    device, output = None, None

    assert eval_year == '2017' or eval_year == '2018' or eval_year == '2019'

    if output is None:
        output = os.path.join("output", "{}_{}_{}".format(eval_year, modelname,
                                                          "pretrained_" + pretrained if pretrained else "random"))
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    bert_out_path = os.path.join(output, (modelname + '_ft' if isFinetune else modelname) + '.res')

    # Initialization model
    device, tokenizer, model = init_model.model_init(device, pretrained, isFinetune, output, path_to_trained_model)

    # Dataset
    dataset = TrecPMDataset(dataset_path, path_to_pos_qrel, path_to_query, collection_path, tokenizer, 'train',
                               num_neg_per_pos=num_negative, bert_k=bert_k)
    dataloaders = {'train': torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                shuffle=True, pin_memory=(device.type == "cuda"),
                                drop_last=True)}

    trainBERT = trainer(model, pretrained, output, batch_size, num_epochs, dataloaders, device, isFinetune, lr)
    trainBERT.train()

    inference(model, tokenizer, dev_collection_path, dev_query_path, dev_dataset_path, dev_qrel_path, bert_out_path, 1)

    trec_eval.eval_set(qrel_out_path, dev_dataset_path, os.path.join(output, ir_method))
    trec_eval.eval_set(qrel_out_path, bert_out_path, os.path.join(output, modelname))


if __name__ == '__main__':
    main()
