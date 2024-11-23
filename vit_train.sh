#!/bin/bash

#SBATCH --job-name=vit_train_imagenet1k
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=20:00:00
#SBATCH --mem=48000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

module load miniconda/3
conda activate vllm
python vit_train_imagenet.py
echo "Done!"