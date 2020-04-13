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
from scipy import misc
from torch import optim, nn
from torch.autograd import Variable
from torchvision.utils import save_image
from vae_gan_net import *
import numpy as np
import pickle
import time
import random
import os
from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from torchvision import transforms

im_size = 128



def process_batch(batch):
    x = torch.from_numpy(np.asarray(batch, dtype=np.float32)).cuda()
    # x = torch.from_numpy(np.asarray(batch, dtype=np.float32) / 255.)
    x = x.view(-1, 1, im_size, im_size)

    return x
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('ConvTranspose2d') != -1:
        m.weight.data.normal_(0.0, 0.02)



def main():
    input_channels = 1
    hidden_size = 128
    max_epochs = 500
    lr = 3e-4

    beta = 20
    alpha = 0.2
    gamma = 30
    batch_size = 60

    G = VAE_GAN_Generator(input_channels, hidden_size).cuda()
    D = Discriminator(input_channels).cuda()

    # G.load_state_dict(torch.load('G.pkl'))
    # D.load_state_dict(torch.load('D.pkl'))
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCELoss()
    criterion.cuda()

    opt_enc = optim.RMSprop(G.encoder.parameters(), lr=lr, alpha=0.9)
    opt_dec = optim.RMSprop(G.decoder.parameters(), lr=lr, alpha=0.9)
    opt_dis = optim.RMSprop(D.parameters(), lr=lr * alpha, alpha=0.9)
    #opt_dis = optim.RMSprop(D.parameters(), lr=lr )
    fixed_noise = Variable(torch.randn(batch_size, hidden_size)).cuda()


    for epoch in range(max_epochs):
        G.train()
        D.train()

        #tmp= epoch % 5
        with open('./data_noise_128.pkl', 'rb') as pkl:
            data_noise = pickle.load(pkl)
        with open('../vae_gan_brain/data_fold_train_128.pkl', 'rb') as pkl:
            data_train = pickle.load(pkl)

        #data_train=data_train[0:13376]
        print("Train set size:", len(data_train))



        batches = batch_provider(data_train, batch_size, process_batch, report_progress=True)
        batches_noise = batch_provider(data_noise, batch_size, process_batch, report_progress=True)

        D_real_list, D_rec_enc_list, D_rec_noise_list, D_list = [], [], [], []
        g_loss_list, rec_loss_list, prior_loss_list = [], [], []

        epoch_start_time = time.time()



        i = 0
        for x_noise,org in zip(batches_noise,batches):
            # ones_label = torch.ones(batch_size).cuda()
            # zeros_label = torch.zeros(batch_size).cuda()
            ones_label =  Variable(torch.ones(batch_size)).cuda()
            zeros_label =  Variable(torch.zeros(batch_size)).cuda()

            datav = Variable(x_noise).cuda()
            orgv=Variable(org).cuda()
            mean, logvar, rec_enc = G(datav)

            noisev = Variable(torch.randn(batch_size, hidden_size)).cuda()
            rec_noise = G.decoder(noisev)
            #
            # ======== Train Discriminator ======== #

            frozen_params(G)
            free_params(D)
            #

            # train discriminator
            output = D(orgv)
            output=output.squeeze(1)
            errD_real = criterion(output, ones_label)
            D_real_list.append(output.data.mean())
            output = D(rec_enc)
            output=output.squeeze(1)
            errD_rec_enc = criterion(output, zeros_label)
            D_rec_enc_list.append(output.data.mean())
            output = D(rec_noise)
            output=output.squeeze(1)
            errD_rec_noise = criterion(output, zeros_label)
            D_rec_noise_list.append(output.data.mean())

            dis_img_loss = errD_real + errD_rec_enc + errD_rec_noise
            #dis_img_loss =  errD_real + errD_rec_enc
           # print ("print (dis_img_loss)", dis_img_loss)
            D_list.append(dis_img_loss.data.mean())
            opt_dis.zero_grad()
            dis_img_loss.backward(retain_graph=True)
            opt_dis.step()
                    # ======== Train Generator ======== #

            free_params(G)
            frozen_params(D)

            # train decoder
            output = D(orgv)
            output=output.squeeze(1)
            errD_real = criterion(output, ones_label)
            output = D(rec_enc)
            output=output.squeeze(1)
            errD_rec_enc = criterion(output, zeros_label)
            output = D(rec_noise)
            output=output.squeeze(1)
            errD_rec_noise = criterion(output, zeros_label)

            similarity_rec_enc = D.similarity(rec_enc)
            similarity_data = D.similarity(orgv)

            dis_img_loss = errD_real + errD_rec_enc + errD_rec_noise
            #dis_img_loss = errD_real + errD_rec_enc
            #print ("dis_img_loss",dis_img_loss)
            #gen_img_loss = - dis_img_loss
            gen_img_loss = -dis_img_loss

            g_loss_list.append(gen_img_loss.data.mean())
            rec_loss = ((similarity_rec_enc - similarity_data) ** 2).mean()
            rec_loss_list.append(rec_loss.data.mean())
            err_dec = gamma * rec_loss + gen_img_loss
            #print("err_dec",err_dec)
            opt_dec.zero_grad()
            err_dec.backward(retain_graph=True)
            opt_dec.step()

            # train encoder
            prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
            prior_loss = (-0.5 * torch.sum(prior_loss)) / torch.numel(mean.data)
            #print (prior_loss, mean, std)
            prior_loss_list.append(prior_loss.data.mean())
            err_enc = prior_loss + beta * rec_loss

            opt_enc.zero_grad()
            err_enc.backward()
            opt_enc.step()




            #############################################
            os.makedirs('results_ori', exist_ok=True)
            os.makedirs('results_rec', exist_ok=True)
            os.makedirs('results_gen', exist_ok=True)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            # report losses and save samples each 60 iterations
            m = 6
            i += 1
            if epoch%5==0 and i % m == 0:
                print(
                    '[%d/%d]: D_real:%.4f, D_enc:%.4f, D_noise:%.4f, Loss_D:%.4f,Loss_G:%.4f, rec_loss:%.4f, prior_loss:%.4f'
 #                   '[%d/%d]: D_real:%.4f, D_enc:%.4f, Loss_D:%.4f, \\'

                    % (epoch,
                       max_epochs,
                       torch.mean(torch.tensor(D_real_list)),
                       torch.mean(torch.tensor(D_rec_enc_list)),
                       torch.mean(torch.tensor(D_rec_noise_list)),
                       torch.mean(torch.tensor(D_list)),
                       torch.mean(torch.tensor(g_loss_list)),
                       torch.mean(torch.tensor(rec_loss_list)),
                       torch.mean(torch.tensor(prior_loss_list))))

                with torch.no_grad():
                   D.eval()
                   G.eval()
                   _, _, x_rec = G.forward(x_noise)
                   x_gen = G.decoder(fixed_noise)
                   x_noise=x_noise.cpu()
                   x_gen=x_gen.cpu()
                   x_rec=x_rec.cpu()


                   # save_image(resultsample.view(-1, 3, im_size, im_size),
                   #            'results_rec/sample_' + str(epoch) + "_" + str(i) + '.png')
                   for j in range(20,29):
                       org_img = transforms.ToPILImage()(x_noise[j].squeeze(0)).convert('L')
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
        del batches_noise
        del data_noise
    print("Training finish!... save training results")
    torch.save(G.state_dict(), "G_noise.pkl")
    torch.save(D.state_dict(), "D_noise.pkl")



if __name__ == '__main__':
    main()
