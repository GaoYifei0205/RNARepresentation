#!/bin/bash

#SBATCH --job-name=preprocess        # 作业名
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mem=30G

python train.py --dataset "sampled_data" --gpu_id 4 --config configs/RNAgraph_graph_classification_GCN_in_vivo_100k.json --debias False --model GCN --fold_algo linearpartition