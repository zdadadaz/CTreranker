#!/bin/bash
python3 train.py --model_name bm25_BERT_txt --pretrained base --year 2019 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1
python3 train.py --model_name bm25_BERT_txt --pretrained base --year 2018 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1
python3 train.py --model_name bm25_BERT_txt --pretrained base --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1
