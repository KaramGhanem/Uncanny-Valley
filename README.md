# The Uncanny Valley: A Comprehensive Analysis of Diffusion Models
Implementation was inspired by code in https://github.com/lucidrains/denoising-diffusion-pytorch.
We conducted our study and built our code based on the structure outlined in the github above.

We have created seperate script files for Ho et al.'s DDPM for different datasets.
We have also created seperate script files for Misguided diffusion and Guided Diffusion.

Most of the scripts have been created previously and are inspired by various studies.
Our main contribution code-wise, happens to be Misguided Diffusion where classifier guidance
is conducted with an untrained classifier. 

Otherwise all variations of Diffusion Models that we have experimented on have various 
open-source implementations.

The scripts from Elucidating the Design Space of Diffusion-Based Generative Models (EDM) by
Karras et al. and the DDPM scripts happen to have different approaches of setting the iterations,
including the vast difference in how both scripts are designed. We took that into considration
and saved checkpoints at comparable iteraitons between both scripts. We used epochs to compare
the training iterations in our paper.

Paper:
https://arxiv.org/pdf/2402.13369.pdf 
