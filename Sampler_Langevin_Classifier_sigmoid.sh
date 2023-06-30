#!/bin/bash

#SBATCH --partition=unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=8                                # Ask for 8 CPUs
#SBATCH --gres=gpu:2                                     # Ask for 2 GPU
#SBATCH --mem=48G                                        # Ask for 48 GB of RAM
#SBATCH -o /network/scratch/k/karam.ghanem/slurm-%j.out  # Write the log on scratch

# 1. Load the required modules
module load miniconda/3 cuda/11.7

# 2. Load your environment
conda activate py3.9

pip install scikit-learn

#pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# srun --mem=8G conda install pytorch cudatoolkit=11.7 -c pytorch -c conda-forge

# 3. Copy your dataset on the compute node
#cp -r /home/mila/k/karam.ghanem/Diffusion/cifar10_png/train /network/datasets/cifar10_train $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
#python /home/mila/k/karam.ghanem/Diffusion/minDiffusion/DDPM.py --channel 3  --save_and_sample_every 5000 --sampling_timesteps 11900 --timesteps 12000 --train_num_steps 20000 --experiment_name Langevin_Sampler_DDIM_cifar10_20k --data_path '/home/mila/k/karam.ghanem/Diffusion/cifar10_png/train' 

python /home/mila/k/karam.ghanem/Diffusion/minDiffusion/DDPM_classifier.py --channel 3  --save_and_sample_every 1 --sampling_timesteps 1000 --timesteps 1000 --experiment_name Langevin_Classifier_sigmoid --data_path '/home/mila/k/karam.ghanem/scratch/Diffusion/cifar10_png/train' --fls_train_path "/home/mila/k/karam.ghanem/scratch/Diffusion/minDiffusion/datasets_cifar_big/cifar_train" --fls_test_path "/home/mila/k/karam.ghanem/scratch/Diffusion/minDiffusion/datasets_cifar_big/cifar_test" --milestone_path "/network/scratch/k/karam.ghanem/Diffusion/minDiffusion/DDPM_sigmoid_schedule_CIFAR10_results"


# 5. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR  /network/scratch/k/karam.ghanem
