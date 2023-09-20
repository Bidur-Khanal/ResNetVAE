#!/bin/bash -l


#SBATCH --account mvaal --partition tier3
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
module purge
conda activate dplearning

python3 -u ResNetVAE.py --root /home/bk9618/learning-with-noisy-labels-benchmark/data/ --dataset-name $dataset --epochs $epochs --batch-size $batch --lr $lr --version $version

