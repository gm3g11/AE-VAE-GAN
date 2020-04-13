# AE-VAE-GAN
Try AE,VAE,DCGAM,AEWGAN, VAEDCGAN on brain dataset in Pytorch

1.Dataset   

2.Model

3.Results  

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

4. Summuray  
VAE_DCGAN could help to denoise and the recovered images are also deblured compared to VAE results.
