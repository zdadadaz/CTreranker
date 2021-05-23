#!/bin/bash
#python3 train.py --model_name bm25_BERT_length256_neg6 --pretrained base --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 6
python3 train_v2.py --model_name bm25_BERT_length256_neg6_v2 --pretrained base --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 6
