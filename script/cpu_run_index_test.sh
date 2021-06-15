#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=jc_run_cpu
#SBATCH -n 1
#SBATCH -w gpunode-1-7
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
srun python3 xml2jsonl_text_v2.py --IPath ../../data/test_collection/clinicaltrials_xml_new/ --OPath ../../data/test_collection/clinicaltrials_json/ --yr 2016
cd /scratch/itee/s4575321/code/pyserini
srun python3 -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 10 -input /scratch/itee/s4575321/data/test_collection/clinicaltrials_json -index indexes/test_collection -storePositions -storeDocvectors -storeRaw