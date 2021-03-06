#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=jc_index_2019
#SBATCH -n 1
#SBATCH -w gpunode-1-8
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
cd /scratch/itee/s4575321/code/cttest
srun python3 xml2jsonl_text_v2.py --IPath ../../data/TRECPM2019/clinicaltrials_xml/ --OPath ../../data/TRECPM2019/clinicaltrials_json_txt_v2/ --yr 2019
cd /scratch/itee/s4575321/code/pyserini
srun python3 -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 10 -input /scratch/itee/s4575321/data/TRECPM2019/clinicaltrials_json_txt_v2 -index indexes/TRECPM2019_txt_v2 -storePositions -storeDocvectors -storeRaw