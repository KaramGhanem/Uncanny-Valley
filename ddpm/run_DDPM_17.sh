#!/bin/bash

#SBATCH --partition=unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=8                                # Ask for 8 CPUs
#SBATCH --gres=gpu:2                                     # Ask for 2 GPU
#SBATCH --mem=48G                                        # Ask for 48 GB of RAM
#SBATCH -o /network/scratch/k/karam.ghanem/slurm-%j.out  # Write the log on scratch

# 1. Load the required modules
module --quiet load miniconda/3

# 2. Load your environment
conda activate py3.8

# 3. Copy your dataset on the compute node
#cp /home/mila/k/karam.ghanem/Diffusion/cifar10_png/train /network/datasets/cifar10_train $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python /home/mila/k/karam.ghanem/Diffusion/minDiffusion/DDPM_distillation.py --channel 1  ---save_and_sample_every 1250 --sampling_timesteps 2000 --timesteps 2000 --train_num_steps 2500  --experiment_name Distillation_Sampler_MNIST_final --data_path '/home/mila/k/karam.ghanem/Diffusion/MNIST Dataset JPG format/MNIST - JPG - testing'

# 5. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR  /network/scratch/k/karam.ghanem

