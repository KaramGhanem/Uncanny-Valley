
#!/bin/bash
#SBATCH --nodes=1-1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=4
#SBATCH --output=myjob_output_wrapper.out
#SBATCH -o /network/scratch/k/karam.ghanem/slurm-%j.out  # Write the log on scratch

# 1. Load the required modules
module load miniconda/3 cuda/11.7

# 2. Load your environment
conda activate py3.8

#3. Load ImageNet Data into temporary folder (Not needed)
# mkdir -p $SLURM_TMPDIR/imagenet/train
# cd       $SLURM_TMPDIR/imagenet/train
# tar  -xf /network/datasets/imagenet/ILSVRC2012_img_train.tar --to-command='mkdir ${TAR_REALNAME%.tar}; tar -xC ${TAR_REALNAME%.tar}'

#ImageNet 64x64 dataset
#saves checkpoint every 5 epochs for 95 epochs (one step is a batch update which is 32 images and there are 1281152 images in the ImageNet training dataset)

srun --gres=gpu:a100:1 -n1 --mem=16G -l --output=%j-step-%s.out --exclusive python /network/scratch/k/karam.ghanem/Diffusion/minDiffusion/DDPM_ImageNet.py --channel 3 --save_and_sample_every 200180  --sampling_timesteps 1000 --timesteps 1000 --experiment_name DDPM_ImageNet_linear_0.25 --train_num_steps 3804420 --data_path '/network/datasets/imagenet.var/imagenet_torchvision/' &
srun --gres=gpu:a100:1 -n1 --mem=16G -l --output=%j-step-%s.out --exclusive python /network/scratch/k/karam.ghanem/Diffusion/minDiffusion/DDPM_ImageNet.py --channel 3 --save_and_sample_every 200180 --sampling_timesteps 1000 --timesteps 1000 --experiment_name DDPM_ImageNet_linear_0.5 --train_num_steps 3804420 --data_path '/network/datasets/imagenet.var/imagenet_torchvision/' &
wait

# 5. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR  /network/scratch/k/karam.ghanem
