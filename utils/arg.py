import argparse

def argretrieve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help="model_name")
    parser.add_argument('--pretrained', help="base, BlueBERT, ClinicalBERT, BioBERT, SciBERT", default='base')
    parser.add_argument('--year', help="choose eval year 2017-2019", default='2019')
    parser.add_argument('--irmethod', help="bm25 or DFR", default='bm25')
    parser.add_argument('--bert_k', help="bert k", default=50)
    parser.add_argument('--lr', help="lr", default=1e-4)
    parser.add_argument('--batch_size', help="batch_size", default=32)
    parser.add_argument('--num_workers', help="num_workers", default=4)
    parser.add_argument('--isFinetune', help="is fine-tune", default=0)
    parser.add_argument('--num_epochs', help="num of epoch", default=1)
    parser.add_argument('--path_to_pretrain', help="path to trained model", default=None)
    parser.add_argument('--num_negative', help="num of negative sample per query", default=10)
    return parser.parse_args()
