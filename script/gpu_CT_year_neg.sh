#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=jc_run_cpu
#SBATCH -n 1
#SBATCH -w gpunode-1-14
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o out.txt
#SBATCH -e erro.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4

module load anaconda/3.6
source activate /scratch/itee/s4575321/env/ct37
module load gnu/5.4.0
module load mvapich2
cd /scratch/itee/s4575321/code/cttest

srun python3 train.py --model_name bm25_BERT_length256_neg14 --pretrained base --year 2019 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 14
srun python3 train.py --model_name bm25_BERT_length256_neg14 --pretrained base --year 2018 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 14
srun python3 train.py --model_name bm25_BERT_length256_neg14 --pretrained base --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 14

srun python3 train.py --model_name bm25_BERT_length256_neg16 --pretrained base --year 2019 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 16
srun python3 train.py --model_name bm25_BERT_length256_neg16 --pretrained base --year 2018 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 16
srun python3 train.py --model_name bm25_BERT_length256_neg16 --pretrained base --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 16

srun python3 train.py --model_name bm25_BERT_length256_neg18 --pretrained base --year 2019 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 18
srun python3 train.py --model_name bm25_BERT_length256_neg18 --pretrained base --year 2018 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 18
srun python3 train.py --model_name bm25_BERT_length256_neg18 --pretrained base --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 18

srun python3 train.py --model_name bm25_BERT_length256_neg20 --pretrained base --year 2019 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 20
srun python3 train.py --model_name bm25_BERT_length256_neg20 --pretrained base --year 2018 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 20
srun python3 train.py --model_name bm25_BERT_length256_neg20 --pretrained base --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 20


#python3 train_v2.py --model_name bm25_BERT_test --pretrained base --year 2019 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1