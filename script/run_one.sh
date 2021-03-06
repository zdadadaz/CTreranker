#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=jc_run_one
#SBATCH -n 1
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o out.txt
#SBATCH -e erro.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2

module load anaconda/3.6
source activate /scratch/itee/s4575321/env/ct37
module load gnu/5.4.0
module load mvapich2
cd /scratch/itee/s4575321/code/cttest
#python utils/mesh_symptom.py
#python utils/ctparser.py
srun python3 test_case.py
#srun python3 xml2jsonl_text_v2.py --IPath ../../data/test_collection/clinicaltrials_xml_new/ --OPath ../../data/test_collection/clinicaltrials_json/ --yr 2016
#srun python3 train_judge.py --model_name bm25_BERT_length256_neg10_all_bm25_none_epoch10        --task all_bm25_none      --pretrained BioBERT --year 2019 --irmethod bm25 --bert_k 50 --lr 2e-5 --isFinetune 1 --num_epochs 10 --num_negative 10

