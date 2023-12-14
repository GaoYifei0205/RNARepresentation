#!/bin/bash

#SBATCH --job-name=preprocess        # 作业名
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mem=30G

python preprocess.py "sampled_data" False linearpartition True