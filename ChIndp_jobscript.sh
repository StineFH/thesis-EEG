#!/bin/bash
#SBATCH --job-name=CHLinearModel
#SBATCH --account= *project name* 
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=small-g
#SBATCH --gpus-per-task=2

srun singularity exec -B /*directory_to_data_onLUMI*/:/data *container name.sif* python run_ChLinearTransformer.py