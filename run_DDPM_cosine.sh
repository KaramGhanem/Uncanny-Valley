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
# python /home/mila/k/karam.ghanem/scratch/Diffusion/minDiffusion/DDPM_ImageNet.py --channel 3 --save_and_sample_every 15687 --train_num_steps 80000 --sampling_timesteps 12000 --timesteps 12000 --experiment_name Langevin_Sampler_ImageNet --data_path '/home/mila/k/karam.ghanem/scratch/datasets/'

# python /home/mila/k/karam.ghanem/scratch/Diffusion/minDiffusion/DDPM.py --channel 3 --save_and_sample_every 625 --train_num_steps 3125 --sampling_timesteps 1000 --timesteps 1000 --experiment_name Langevin_Sampler_linear_schedule_CIFAR10 --beta_schedule 'linear' --data_path '/home/mila/k/karam.ghanem/scratch/Diffusion/cifar10_png/train'

# python /home/mila/k/karam.ghanem/scratch/Diffusion/minDiffusion/DDPM.py --channel 3 --save_and_sample_every 625 --train_num_steps 3125 --sampling_timesteps 1000 --timesteps 1000 --experiment_name Langevin_Sampler_cosine_schedule_CIFAR10 --beta_schedule 'sigmoid' --data_path '/home/mila/k/karam.ghanem/scratch/Diffusion/cifar10_png/train'

python /home/mila/k/karam.ghanem/scratch/Diffusion/minDiffusion/DDPM.py --channel 3 --save_and_sample_every 15625 --train_num_steps 156250 --sampling_timesteps 1000 --timesteps 1000 --experiment_name Langevin_Sampler_sigmoid_schedule_CIFAR10 --beta_schedule 'cosine' --data_path '/home/mila/k/karam.ghanem/scratch/Diffusion/cifar10_png/train'

# 5. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR  /network/scratch/k/karam.ghanem


