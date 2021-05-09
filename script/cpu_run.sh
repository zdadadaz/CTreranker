#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=jc_run_cpu
#SBATCH -n 1
#SBATCH -w gpunode-1-14
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o out.txt
#SBATCH -e erro.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

module load anaconda/3.6
source activate /scratch/itee/s4575321/env/ct37
module load gnu/5.4.0
module load mvapich2
cd /scratch/itee/s4575321/code/ct_reranker
# srun python3 -m pip freeze
# srun python3 simpletest.py
srun python3 xml2jsonl.py
cd /scratch/itee/s4575321/code/pyserini
srun python3 -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 4 -input /scratch/itee/s4575321/data/TRECPM2019/clinicaltrials_json_bt -index indexes/TRECPM -storePositions -storeDocvectors -storeRaw