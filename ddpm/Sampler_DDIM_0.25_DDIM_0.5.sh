#!/bin/bash
#SBATCH --nodes=1-1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=6
#SBATCH --output=myjob_output_wrapper.out
#SBATCH -o /network/scratch/k/karam.ghanem/slurm-%j.out  # Write the log on scratch

# 1. Load the required modules
module load miniconda/3 cuda/11.7

# 2. Load your environment
conda activate py3.8

srun --gres=gpu:a100:1 -n1 --mem=16G -l --output=%j-step-%s.out --exclusive python /network/scratch/k/karam.ghanem/Diffusion/minDiffusion/DDPM.py --channel 3  --save_and_sample_every 1 --sampling_timesteps 999 --timesteps 1000 --experiment_name DDIM_linear_0.25 --data_path '/home/mila/k/karam.ghanem/scratch/Diffusion/cifar10_png/train' --fls_train_path "/home/mila/k/karam.ghanem/scratch/Diffusion/minDiffusion/datasets_cifar_big/cifar_train" --fls_test_path "/home/mila/k/karam.ghanem/scratch/Diffusion/minDiffusion/datasets_cifar_big/cifar_test" --milestone_path "/network/scratch/k/karam.ghanem/Diffusion/minDiffusion/DDPM_linear_0.25_schedule_CIFAR10_results" &
srun --gres=gpu:a100:1 -n1 --mem=16G -l --output=%j-step-%s.out --exclusive python /network/scratch/k/karam.ghanem/Diffusion/minDiffusion/DDPM.py --channel 3  --save_and_sample_every 1 --sampling_timesteps 999 --timesteps 1000 --experiment_name DDIM_linear_0.5 --data_path '/home/mila/k/karam.ghanem/scratch/Diffusion/cifar10_png/train' --fls_train_path "/home/mila/k/karam.ghanem/scratch/Diffusion/minDiffusion/datasets_cifar_big/cifar_train" --fls_test_path "/home/mila/k/karam.ghanem/scratch/Diffusion/minDiffusion/datasets_cifar_big/cifar_test" --milestone_path "/network/scratch/k/karam.ghanem/Diffusion/minDiffusion/DDPM_linear_0.5_schedule_CIFAR10_results" &
wait


# 5. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR  /network/scratch/k/karam.ghanem

