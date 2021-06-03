#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=jc_judge
#SBATCH -n 1
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o out.txt
#SBATCH -e ite_erro.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

module load anaconda/3.6
source activate /scratch/itee/s4575321/env/ct37
module load gnu/5.4.0
module load mvapich2
cd /scratch/itee/s4575321/code/cttest

#srun python3 train_judge_iteration.py --model_name bm25_BERT_length256_neg10_judged_bm25_bert_epoch10        --task judged_bm25_bert      --pretrained BioBERT --year 2019 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 10 --num_negative 10
srun python3 train_judge_iteration.py --model_name bm25_BERT_length256_neg10_all_bert_none_epoch10        --task all_bert_none      --pretrained BioBERT --year 2019 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 10 --num_negative 10

#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_judged_random_bm25     --task judged_random_bm25        --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_judged_random_otjudged --task judged_random_otjudged    --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_judged_bm25_bm25     --task judged_bm25_bm25      --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_judged_bm25_otjudged --task judged_bm25_otjudged  --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_judged_bert_bm25     --task judged_bert_bm25     --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_judged_bert_otjudged --task judged_bert_otjudged --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_unjudged_random_none --task unjudged_random_none  --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_unjudged_bm25_none   --task unjudged_bm25_none    --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_unjudged_bert_none   --task unjudged_bert_none    --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_all_random_none      --task all_random_none    --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_all_bm25_none        --task all_bm25_none      --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_all_bert_none        --task all_bert_none      --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10




#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_judged_random   --task judged_random    --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_judged_bm25     --task judged_bm25      --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_judged_bert     --task judged_bert      --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_unjudged_random --task unjudged_random  --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_unjudged_bm25   --task unjudged_bm25    --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_unjudged_bert   --task unjudged_bert    --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_all_random      --task all_random    --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_all_bm25        --task all_bm25      --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_all_bert        --task all_bert      --pretrained BioBERT --year 2017 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 1 --num_negative 10

