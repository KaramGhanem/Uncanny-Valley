#!/bin/bash

#SBATCH --partition=unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=8                                # Ask for 8 CPUs
#SBATCH --gres=gpu:2                                     # Ask for 2 GPU
#SBATCH --mem=48G                                        # Ask for 48 GB of RAM
#SBATCH -o /network/scratch/k/karam.ghanem/slurm-%j.out  # Write the log on scratch

# 1. Load the required modules
module --quiet load miniconda/3

# 2. Load your environment
conda activate edm
module load cuda/11.7

# 3. Copy your dataset on the compute node
#cp /home/mila/k/karam.ghanem/Diffusion/cifar10_png/train /network/datasets/cifar10_train $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR

#Imagenet

#CIFAR10

#train
python /network/scratch/k/karam.ghanem/Diffusion/minDiffusion/score_sde_pytorch/main.py --config /network/scratch/k/karam.ghanem/Diffusion/minDiffusion/score_sde_pytorch/configs/ve/ncsn/cifar10.py --eval_folder /network/scratch/k/karam.ghanem/Diffusion/minDiffusion/score_sde_pytorch/ncsn_results --mode "train" --workdir /network/scratch/k/karam.ghanem/Diffusion/minDiffusion/score_sde_pytorch/ncsn_results

#Sample and evaluate P.S. configure enable_sampling to True in the config file

# python /network/scratch/k/karam.ghanem/Diffusion/minDiffusion/score_sde_pytorch/main.py --config /network/scratch/k/karam.ghanem/Diffusion/minDiffusion/score_sde_pytorch/configs/ve/ncsn/cifar10.py --eval_folder /network/scratch/k/karam.ghanem/Diffusion/minDiffusion/score_sde_pytorch/ncsn_results --mode "eval" --workdir /network/scratch/k/karam.ghanem/Diffusion/minDiffusion/score_sde_pytorch/ncsn_results

# 5. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR  /network/scratch/k/karam.ghanem


