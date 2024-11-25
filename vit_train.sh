#!/bin/bash

#SBATCH --job-name=vit_train_imagenet1k
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=20:00:00
#SBATCH --mem=48000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

# Exit immediately if a command exits with a non-zero status
set -e

module load miniconda/3
conda activate vllm

# Download the ImageNet training and validation datasets
echo "Downloading ImageNet training data..."
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate

echo "Downloading ImageNet validation data..."
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate

# Extract the downloaded files
echo "Extracting ImageNet data..."
bash extract_ILSVRC.sh

# Install the required Python package
echo "Installing einops Python library..."
pip install einops

# Run the first training script
echo "Starting training vit_train_imagenet.py..."
python vit_train_imagenet.py

# Run the next script
echo "vit_train_imagenet.py completed successfully. Running nvit_train.sh..."
bash nvit_train.sh

echo "Done!"