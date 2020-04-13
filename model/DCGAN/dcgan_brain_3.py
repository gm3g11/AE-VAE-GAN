import argparse
import os
import numpy as np
import math
import pickle
from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images_3", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=60, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            #[B,128,32,32]
            nn.BatchNorm2d(128),
            #nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            # [B,128,64,64]
            #nn.Conv2d(128, 128, 3, stride=1, padding=1),
            # [B,128,64,64]
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(64, 32, 4,stride=2, padding=1),
            # [B,128,128,128]
            #nn.Conv2d(128, 64, 3, stride=1, padding=1),
            # [B,64,128,128]
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1),
            # [B,128,128,128]
            # nn.Conv2d(128, 64, 3, stride=1, padding=1),
            # [B,64,128,128]
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv2d(16, opt.channels, 3, stride=1, padding=1),
            # [B,1,128,128]
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img # [B,1,128,128]


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            # in_filters: [B,1,128,128]
            # out_filters: [B,1]
            block = [nn.Conv2d(in_filters, out_filters, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            # [B,1,128,128]
            *discriminator_block(opt.channels, 16, bn=False),
            # [B,16,64,64]
            *discriminator_block(16, 32),
            # [B,32,32,32]
            *discriminator_block(32, 64),
            # [B,64,16,16]
            *discriminator_block(64, 128),
            # [B,128,8,8]
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4 #128//(2^4)=8
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        # [B,128,8,8]
        out = out.view(out.shape[0], -1)
        # [B,128x8x8]
        validity = self.adv_layer(out)
        #[B,1]

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
# os.makedirs("./data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "./data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
def process_batch(batch):
    x = torch.from_numpy(np.asarray(batch, dtype=np.float32)).cuda()
    # x = torch.from_numpy(np.asarray(batch, dtype=np.float32) / 255.)
    x = x.view(-1, 1, opt.img_size,opt.img_size)

    return x

for epoch in range(opt.n_epochs):
    i=0
    with open('./data_fold_train_128.pkl', 'rb') as pkl:
        data_train = pickle.load(pkl)
    batches = batch_provider(data_train, opt.batch_size, process_batch, report_progress=True)
    for imgs in batches:
        i=i+1

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(batches), d_loss.item(), g_loss.item())
        )


        if epoch % 10 == 0:
            #save_image(gen_imgs.data[:25], "images/%d.png" % epoch, nrow=5, normalize=True)
            save_image(gen_imgs.data[:1], "images_3/%d.png" % epoch, nrow=1)
