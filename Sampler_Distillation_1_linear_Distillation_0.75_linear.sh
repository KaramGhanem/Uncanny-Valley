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

srun --immediate --exact --gres=gpu:a100:1 -n1 -c 4 --mem=16G -l --output=%j-step-%s.out python /network/scratch/k/karam.ghanem/Diffusion/minDiffusion/DDPM_distillation.py --channel 3  --save_and_sample_every 1 --sampling_timesteps 1000 --timesteps 1000 --experiment_name Distillation_linear_1 --data_path '/home/mila/k/karam.ghanem/scratch/Diffusion/cifar10_png/train' --fls_train_path "/network/scratch/k/karam.ghanem/Diffusion/minDiffusion/datasets_cifar_big/cifar_train" --fls_test_path "/network/scratch/k/karam.ghanem/Diffusion/minDiffusion/datasets_cifar_big/cifar_test" --milestone_path "/network/scratch/k/karam.ghanem/Diffusion/minDiffusion/DDPM_linear_1_schedule_CIFAR10_results" &
srun --immediate --exact --gres=gpu:a100:1 -n1 -c 4 --mem=16G -l --output=%j-step-%s.out python /network/scratch/k/karam.ghanem/Diffusion/minDiffusion/DDPM_distillation.py --channel 3  --save_and_sample_every 1 --sampling_timesteps 1000 --timesteps 1000 --experiment_name Distillation_linear_0.75 --data_path '/home/mila/k/karam.ghanem/scratch/Diffusion/cifar10_png/train' --fls_train_path "/network/scratch/k/karam.ghanem/Diffusion/minDiffusion/datasets_cifar_big/cifar_train" --fls_test_path "/network/scratch/k/karam.ghanem/Diffusion/minDiffusion/datasets_cifar_big/cifar_test" --milestone_path "/network/scratch/k/karam.ghanem/Diffusion/minDiffusion/DDPM_linear_0.75_schedule_CIFAR10_results" &
wait


# 5. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR  /network/scratch/k/karam.ghanem

