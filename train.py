
from transformers import BertForSequenceClassification
from utils.tradquery import Tradquery
from utils.gendata import Dataset
from utils import readfile as rf
from eval import trec_eval as eval
import torch
import os
import pathlib
from sklearn.model_selection import train_test_split
from utils.training import trainer
import argparse

def argretrieve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help="model_name")
    parser.add_argument('--pretrained', help="pretrain")
    parser.add_argument('--year', help="choose eval year 2017-2019")
    parser.add_argument('--irmethod', help="bm25 or DFR", default='bm25')
    parser.add_argument('--bert_k', help="bert k", default=50)
    parser.add_argument('--lr_step_period', help="pretrain",default=None)
    parser.add_argument('--batch_size', help="batch_size", default=32)
    parser.add_argument('--num_workers', help="num_workers", default=4)
    return parser.parse_args()


def main():
    # modelname = 'bm25_BERT', pretrained = 'base', lr_step_period = None, output = None
    root_path = '../../data/TRECPM2017'
    indexing_path = '../pyserini/indexes/TRECPM2019'
    args = argretrieve()
    modelname = args.model_name
    pretrained = args.pretrained
    lr_step_period = args.lr_step_period
    batch_size = args.batch_size #2
    num_workers= args.num_workers #1

    device, output = None, None

    num_epochs = 1
    eval_year = args.year #'2019'
    IR_method = args.irmethod#'bm25'
    bert_k = int(args.bert_k)#50
    bm25_k = 1000

    assert eval_year == '2017' or eval_year == '2018' or eval_year == '2019'
    assert IR_method == 'bm25' or IR_method == 'DFR'

    # need combine 2017-2019 as a whole query
    topics_nums_by_year = [29,50,38]
    topics =[]
    qrels = []
    for i in ['2017','2018','2019']:
        qrels.append(rf.read_qrel(root_path + '/qrels_{}.txt'.format(i)))
        topics.append(rf.read_topics(root_path + '/topics_{}.xml'.format(i)))
    # delete topic without relevant doc
    del topics[0]['10'] #2017
    del topics[2]['32'] #2019
    del topics[2]['33'] #2019
    query_dict = rf.concat_topics(topics)
    qrel_dict = rf.concat_topics(qrels)

    if output is None:
        output = os.path.join("output","{}_{}_{}".format(modelname, eval_year, "pretrained_"+pretrained if pretrained else "random"))
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    # Initialization model
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
    model.classifier.bias.data[0] = 0.5

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    # train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=1)
    # train 17, 18, test 19
    val_idx = []
    # topics_nums_by_year = [[1, 29], [30, 79], [80, 117]]
    # qbyyear = [[1, 29], [30, 79], [80, 117]]
    if eval_year == '2019':
        train_idx = [str(i) for i in range(1,21)] #81
        test_idx = [str(i) for i in range(80,91)] # 121
    elif eval_year == '2018':
        train_idx = [str(i) for i in range(1, 11) ] +  [str(i) for i in range(80, 91)] # 81
        test_idx = [str(i) for i in range(30, 41)]  # 121
    else: # 2017
        train_idx = [str(i) for i in range(30, 41)] + [str(i) for i in range(80, 91)]  # 81
        test_idx = [str(i) for i in range(1, 11)]  # 121

    dataidx = {'train': train_idx,'val':val_idx,'test':test_idx}
    dataloaders = {}
    # BM25
    for phase in dataidx:
        if len(dataidx[phase]) == 0:
            continue
        bm25 = Tradquery(query_dict, dataidx[phase], indexing_path, output,bm25_k, bert_k, IR_method, phase)
        dataset = Dataset(dataidx[phase], query_dict, qrel_dict, bm25.topklist, bm25.fields, bm25.searcher, phase)
        if phase == 'test':
            dataloaders[phase] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                             pin_memory=(device.type == "cuda"))
        else:
            dataloaders[phase] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                             shuffle=True,pin_memory=(device.type == "cuda"), drop_last=True)
    trainBERT = trainer(model, output, batch_size, num_epochs, dataloaders, device)
    trainBERT.train()
    test_out_path = os.path.join(output, 'pyserini_dev_demofilter_{}_{}_{}.res'.format(IR_method,str(bm25_k), 'test'))
    qrel_out_path = '/'.join(test_out_path.split('/')[:-1])+'/test_qrels.txt'
    trainBERT.test(dataidx['test'], qrel_dict, test_out_path, qrel_out_path, bert_k, IR_method, modelname)
    eval.trec_eval(qrel_out_path, test_out_path)

if __name__ == '__main__':
    main()