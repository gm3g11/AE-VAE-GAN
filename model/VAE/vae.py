# Copyright 2019 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function
import torch.utils.data
#from scipy import misc
from torch import optim
from torchvision.utils import save_image
from net import *
import numpy as np
import pickle
import time
import random
import os
from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from torchvision import transforms
from torch.nn import functional as F
im_size = 128


def loss_function(recon_x, x, mu, logvar):
    #BCE = torch.mean((recon_x - x)**2)
    BCE = F.binary_cross_entropy(recon_x.view(-1, recon_x.shape[2]*recon_x.shape[3]), x.view(-1, x.shape[2]*x.shape[3]), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD * 0.1


def process_batch(batch):

    x = torch.from_numpy(np.asarray(batch, dtype=np.float32)).cuda()

    x = x.view(-1, 1, im_size, im_size)
    #print(x.shape)
    return x


def main():
    batch_size = 60
    #z_size = 512
    z_size = 100
    vae = VAE(zsize=z_size, layer_count=5,channels=1)
    #vae=nn.DataParallel(vae)
    vae.cuda()
    vae.train()
    vae.weight_init(mean=0, std=0.02)

    lr = 0.0005

    vae_optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
 
    train_epoch =1000

    sample1 = torch.randn(batch_size , z_size).view(-1, z_size, 1, 1)
    BCE_list=[]
    KLD_list=[]
    noise_list=[]

    for epoch in range(train_epoch):
        vae.train()
        #tmp= epoch % 5
        with open('../vae_gan_brain/data_fold_train_128.pkl', 'rb') as pkl:
            data_train = pickle.load(pkl)

        # with open('./data_fold_train%d.pkl' % ( (tmp+1) % 5), 'rb') as pkl:
        #     data_train.extend( pickle.load(pkl))
        # with open('./data_fold_train%d.pkl' % ( (tmp+2) % 5), 'rb') as pkl:
        #     data_train.extend( pickle.load(pkl))




        print("Train set size:", len(data_train))

        random.shuffle(data_train)

        batches = batch_provider(data_train, batch_size, process_batch, report_progress=True)

        rec_loss = 0
        kl_loss = 0

        epoch_start_time = time.time()

        if (epoch + 1) % 8 == 0:
            vae_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        i = 0
        for x in batches:
            i=i+1
            vae.train()
            vae.zero_grad()
            rec, mu, logvar,latent_space = vae(x)

            loss_re, loss_kl = loss_function(rec, x, mu, logvar)
            (loss_re + loss_kl).backward()
            vae_optimizer.step()
            rec_loss += loss_re.item()
            kl_loss += loss_kl.item()


            #############################################
            os.makedirs('results_ori', exist_ok=True)
            os.makedirs('results_rec', exist_ok=True)
            os.makedirs('results_gen', exist_ok=True)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            # report losses and save samples each 60 iterations
            if epoch > 750:
                noise_list.append(latent_space )

            if epoch%20==0 and i == 5:
            #if epoch  == 0 and i==1:
                rec_loss /= i
                kl_loss /= i
                print('\n[%d/%d] - ptime: %.2f, rec loss: %.9f, KL loss: %.9f' % (
                    (epoch + 1), train_epoch, per_epoch_ptime, rec_loss, kl_loss))
                rec_loss = 0
                kl_loss = 0
                with torch.no_grad():
                    vae.eval()
                    x_rec, _, _ ,z= vae(x)
                    x_gen = vae.decode(sample1)
                    x=x.cpu()
                    x_gen=x_gen.cpu()
                    x_rec=x_rec.cpu()


                    # save_image(resultsample.view(-1, 3, im_size, im_size),
                    #            'results_rec/sample_' + str(epoch) + "_" + str(i) + '.png')
                    for j in range(20,29):
                        org_img = transforms.ToPILImage()(x[j].squeeze(0)).convert('L')
                        rec_img = transforms.ToPILImage()(x_rec[j].squeeze(0)).convert('L')
                        gen_img = transforms.ToPILImage()(x_gen[j].squeeze(0)).convert('L')
                        org_img.save('results_ori/ori_' + str(epoch) + "_" + str(i) +"_"+str(j)+ '.png')
                        rec_img.save('results_rec/rec_' + str(epoch) + "_" + str(i) + "_"+str(j)+ '.png')
                        gen_img.save('results_gen/gen_' + str(epoch) + "_" + str(i) +"_"+str(j)+  '.png')

                    # resultsample = x_rec * 0.5 + 0.5
                    # resultsample = resultsample.cpu()
                    # save_image(resultsample.view(-1, 3, im_size, im_size),
                    #            'results_gen/sample_' + str(epoch) + "_" + str(i) + '.png')


        del batches
        del data_train
    print("Training finish!... save training results")
    output_latent_space = open('./latent_space.pkl', 'wb')
    # output_bce = open('./BCE_loss.pkl', 'wb')
    pickle.dump(noise_list, output_latent_space)
    # pickle.dump(BCE_list, output_bce)
    output_latent_space.close()
    # output_bce.close()
    torch.save(vae.state_dict(), "VAEmodel.pkl")


if __name__ == '__main__':
    main()
