from PIL import Image
import os

def make_grid(image_paths, grid_size, image_size):
    # Create a new blank image with a white background
    grid_img = Image.new('RGB', (grid_size[0] * image_size[0], grid_size[1] * image_size[1]), color='white')
    
    # Iterate over the grid
    for i, img_path in enumerate(image_paths):
        # Open the image and resize it
        img = Image.open(img_path).resize(image_size)
        # Compute the position where the image will be placed in the grid
        x = i % grid_size[0] * image_size[0]
        y = i // grid_size[0] * image_size[1]
        # Paste the image into the grid
        grid_img.paste(img, (x, y))
    
    return grid_img

#For FFHQ : Creates one Grid per checkpoint

#Gets the same set of images i.e. 000001 - 000100
root_dir = '/home/mila/k/karam.ghanem/scratch/Diffusion/minDiffusion/edm/training-runs/00024-cifar10-32x32-uncond-ncsnpp-edm-gpus1-batch32-fp32/ninth_checkpoint/'

# Size of the grid (number of images horizontally by number of images vertically)
grid_size = (10, 10)  # for example, for a 10x10 grid
# List of image paths
image_paths = [os.path.join(root_dir, f'{str(i).zfill(6)}.png') for i in range(grid_size[0] * grid_size[1])]

# Size of each image in the grid
image_size = (32, 32)  # assuming each image is 64x64 pixels

# Create the grid image
grid_image = make_grid(image_paths, grid_size, image_size)

# Save the grid image
grid_image.save('/home/mila/k/karam.ghanem/scratch/Diffusion/minDiffusion/edm/training-runs/image_grid.png')

#For CIFAR10 : Progressive plotting

def get_image_paths(root_dir, checkpoints, image_name, runs):
    image_paths = []
    for run in runs:
        for checkpoint in checkpoints:
            checkpoint_dir = os.path.join(root_dir, f'run{run}', checkpoint)
            image_path = os.path.join(checkpoint_dir, image_name)
            image_paths.append(image_path)
    return image_paths

# Common settings
image_name = 'specific_image.png'  # Replace with the specific image name you want to pick
checkpoints = [f'checkpoint{i}' for i in range(1, 10)]  # Checkpoints 1 to 9
root_dir = '/home/mila/k/karam.ghanem/scratch/Diffusion/minDiffusion/edm/training-runs'

# Settings for DDPM
runs_DDPM = ['1', '2', '3', '4', '5', '6']  # Runs for DDPM
grid_size_DDPM = (9, 6)
image_paths_DDPM = get_image_paths(root_dir, checkpoints, image_name, runs_DDPM)
grid_image_DDPM = make_grid(image_paths_DDPM, grid_size_DDPM, image_size)

# Save the DDPM grid image
grid_image_DDPM.save('/home/mila/k/karam.ghanem/scratch/Diffusion/minDiffusion/edm/training-runs/ddpm_image_grid.png')

# Settings for SDE
runs_SDE = ['1', '2', '3', '4', '5', '6']  # Runs for SDE
grid_size_SDE = (9, 4)
image_paths_SDE = get_image_paths(root_dir, checkpoints, image_name, runs_SDE)
grid_image_SDE = make_grid(image_paths_SDE, grid_size_SDE, image_size)

# Save the SDE grid image
grid_image_SDE.save('/home/mila/k/karam.ghanem/scratch/Diffusion/minDiffusion/edm/training-runs/sde_image_grid.png')