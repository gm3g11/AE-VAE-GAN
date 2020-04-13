import pickle
import random
import torch.utils.data
import torch
import os
import torch.nn as nn
import torch.optim as optim
from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from tqdm import tqdm

from torch.autograd import Variable

from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
import numpy as np

torch.manual_seed(123)

batch_size=60
epochs=10000
lr=0.0001
dim_h=128
n_z=100
LAMBDA=10
n_channel=1
sigma=1.
im_size=128
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.n_channel =n_channel
        self.dim_h =dim_h
        self.n_z = n_z

        self.main = nn.Sequential(
            #[B,1,128,128]
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h ),
            nn.ReLU(True),
            # [B,128,64,64]
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            # [B,256,32,32]
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            # [B,512,16,16]
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True)
            # [B,1024,8,8]


        )
        self.fc = nn.Linear(self.dim_h * 8 *8*8, self.n_z)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.shape[0], self.dim_h * 8 *8*8)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.n_channel = n_channel
        self.dim_h = dim_h
        self.n_z = n_z

        self.proj = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 8 *8),
            nn.ReLU()
        )

        self.main = nn.Sequential(
            #[B,128*8,8,8]
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4,2,1),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            # [B,512,16,16]
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            # [B,256,32,32]
            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h * 1, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 1),
            nn.ReLU(True),
            # [B,128,64,64]
            nn.ConvTranspose2d(self.dim_h ,1, 4,2,1),
            # [B,1,128,128]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.view(x.shape[0], self.dim_h*8,8,8)
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.n_channel = n_channel
        self.dim_h = dim_h
        self.n_z = n_z

        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x




encoder, decoder, discriminator = Encoder(), Decoder(), Discriminator()
criterion = nn.MSELoss()

encoder.train()
decoder.train()
discriminator.train()

# Optimizers
enc_optim = optim.Adam(encoder.parameters(), lr = lr)
dec_optim = optim.Adam(decoder.parameters(), lr = lr)
dis_optim = optim.Adam(discriminator.parameters(), lr = 0.5 * lr)

enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)
dis_scheduler = StepLR(dis_optim, step_size=30, gamma=0.5)

if torch.cuda.is_available():
    encoder, decoder, discriminator = encoder.cuda(), decoder.cuda(), discriminator.cuda()

one = torch.tensor(1,dtype=torch.float)
mone = one * -1

if torch.cuda.is_available():
    one = one.cuda()
    mone = mone.cuda()



def process_batch(batch):

    x = torch.from_numpy(np.asarray(batch, dtype=np.float32)).cuda()

    x = x.view(-1, 1, im_size, im_size)
    #print(x.shape)
    return x

sample1 = torch.randn(batch_size, n_z).cuda()
for epoch in range(epochs):
    step = 0
    with open('../vae_gan_brain/data_fold_train_128.pkl', 'rb') as pkl:
        data_train = pickle.load(pkl)
    #random.shuffle(data_train)

    batches = batch_provider(data_train, batch_size, process_batch, report_progress=True)

    for images in tqdm(batches):

        if torch.cuda.is_available():
            images = images.cuda()

        encoder.zero_grad()
        decoder.zero_grad()
        discriminator.zero_grad()

        # ======== Train Discriminator ======== #

        frozen_params(decoder)
        frozen_params(encoder)
        free_params(discriminator)

        z_fake = torch.randn(images.size()[0], n_z) *sigma

        if torch.cuda.is_available():
            z_fake = z_fake.cuda()

        d_fake = discriminator(z_fake)

        z_real = encoder(images)
        d_real = discriminator(z_real)

        torch.log(d_fake).mean().backward(mone)
        torch.log(1 - d_real).mean().backward(mone)

        dis_optim.step()

        # ======== Train Generator ======== #

        free_params(decoder)
        free_params(encoder)
        frozen_params(discriminator)

        batch_size = images.size()[0]

        z_real = encoder(images)
        x_recon = decoder(z_real)
        d_real = discriminator(encoder(Variable(images.data)))

        recon_loss = criterion(x_recon, images)
        d_loss = LAMBDA * (torch.log(d_real)).mean()

        recon_loss.backward(one)
        d_loss.backward(mone)

        enc_optim.step()
        dec_optim.step()

        step += 1

        if (step + 1) % 6 == 0 and epoch %10 ==0:
            print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f" %
                  (epoch + 1, epochs, step + 1, len(batches), recon_loss.data.item()))

        if  epoch  % 20 == 0 and step ==5:
            os.makedirs('results_ori', exist_ok=True)
            os.makedirs('results_rec', exist_ok=True)
            os.makedirs('results_gen', exist_ok=True)
            with torch.no_grad():
                encoder.eval()
                decoder.eval()
                discriminator.eval()


                z_real = encoder(Variable(images))
                reconst = decoder(Variable(z_real))
                samp_img=decoder(Variable(sample1))
                images=images.cpu()
                reconst =reconst.cpu()
                samp_img =samp_img.cpu()

                for j in range(20, 29):
                    org_img = transforms.ToPILImage()(images[j].squeeze(0)).convert('L')
                    rec_img = transforms.ToPILImage()(reconst[j].squeeze(0)).convert('L')
                    gen_img = transforms.ToPILImage()(samp_img[j].squeeze(0)).convert('L')
                    org_img.save('results_ori/ori_' + str(epoch) + "_"  + str(j) + '.png')
                    rec_img.save('results_rec/rec_' + str(epoch) + "_"   + str(j) + '.png')
                    gen_img.save('results_gen/gen_' + str(epoch) + "_"  + str(j) + '.png')




    del batches
    del data_train

print("Training finish!... save training results")