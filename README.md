# AE-VAE-GAN
Try AE,VAE,DCGAM,AEWGAN, VAEDCGAN on brain dataset in Pytorch

1.Dataset Introduction

2.Structure

3.results

4.Tricks for training
a.In the Discriminator or Generater/Decoder, uses sigmoid instead of tanh
b.When training the Discriminator, it would be better to freeze Generator weights update and vice verse.
e.g. 
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False
        
frozen_params(G)
free_params(D)

c. The network should be designed deliberately. If too shallow, the image is blurred. If too deep the image is almost black and couldn't see the brain.
