#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=jc_run_v2
#SBATCH -n 1
#SBATCH -w gpunode-0-16
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

srun python3 train_v2.py --model_name bm25_BERT_length256_neg6 --pretrained base --year 2019 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 6
srun python3 train_v2.py --model_name bm25_BERT_length256_neg6 --pretrained base --year 2018 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 6
srun python3 train_v2.py --model_name bm25_BERT_length256_neg6 --pretrained base --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 6

srun python3 train_v2.py --model_name bm25_BERT_length256_neg6 --pretrained BlueBERT --year 2019 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 6
srun python3 train_v2.py --model_name bm25_BERT_length256_neg6 --pretrained BlueBERT --year 2018 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 6
srun python3 train_v2.py --model_name bm25_BERT_length256_neg6 --pretrained BlueBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 6

srun python3 train_v2.py --model_name bm25_BERT_length256_neg6 --pretrained BioBERT --year 2019 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 6
srun python3 train_v2.py --model_name bm25_BERT_length256_neg6 --pretrained BioBERT --year 2018 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 6
srun python3 train_v2.py --model_name bm25_BERT_length256_neg6 --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 6

srun python3 train_v2.py --model_name bm25_BERT_length256_neg6 --pretrained ClinicalBERT --year 2019 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 6
srun python3 train_v2.py --model_name bm25_BERT_length256_neg6 --pretrained ClinicalBERT --year 2018 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 6
srun python3 train_v2.py --model_name bm25_BERT_length256_neg6 --pretrained ClinicalBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 6

srun python3 train_v2.py --model_name bm25_BERT_length256_neg6 --pretrained SciBERT --year 2019 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 6
srun python3 train_v2.py --model_name bm25_BERT_length256_neg6 --pretrained SciBERT --year 2018 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 6
srun python3 train_v2.py --model_name bm25_BERT_length256_neg6 --pretrained SciBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 6


#python3 train_v2.py --model_name bm25_BERT_test --pretrained base --year 2019 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1
python3 train.py --model_name bm25_BERT_length256_neg10_test --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
