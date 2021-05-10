## How to use
* pip install -r requirement
* install pyserini and anserini for BM25 and indexing  

## Train classifier
* python3 train.py --model_name bm25_BERT --pretrained base --year 2019 --irmethod bm25 --bert_k 50 --lr 1e-4 --isFinetune 0 --num_epochs 20

## Fine-tune
* python3 train.py --model_name bm25_BERT --pretrained base --year 2019 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1