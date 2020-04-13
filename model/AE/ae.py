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

im_size = 128


# def loss_function(recon_x, x):
#     BCE = torch.mean((recon_x - x)**2)
#
#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     #KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
#     return BCE


def process_batch(batch):
    #data = [misc.imresize(x, [im_size, im_size]).transpose((2, 0, 1)) for x in batch]

    #x = np.asarray(batch, dtype=np.float32)/255.
    #x = torch.from_numpy(np.asarray(batch, dtype=np.float32))/ 127.5 - 1.
    #data=[y for y in batch]
    x = torch.from_numpy(np.asarray(batch, dtype=np.float32)).cuda()
    # transform = transforms.Compose([
    #     #transforms.ToTensor(),
    #     transforms.Normalize([0.5], [0.5])
    # ]
    # )
    # x=transform(x).cuda()
    x = x.view(-1, 1, im_size, im_size)
    #print(x.shape)
    return x

loss_fn = torch.nn.MSELoss( reduction='mean')
def main():
    batch_size = 60
    #z_size = 512
    z_size = 100
    ae = AE(zsize=z_size, layer_count=5,channels=1)
    #vae=nn.DataParallel(vae)
    ae.cuda()
    ae.train()
    ae.weight_init(mean=0, std=0.02)

    lr = 0.0005

    ae_optimizer = optim.Adam(ae.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
 
    train_epoch =1000


    for epoch in range(train_epoch):
        ae.train()
        #tmp= epoch % 5
        with open('../vae_gan_brain/data_fold_train_128.pkl', 'rb') as pkl:
            data_train = pickle.load(pkl)




        print("Train set size:", len(data_train))

        random.shuffle(data_train)

        batches = batch_provider(data_train, batch_size, process_batch, report_progress=True)

        rec_loss = 0


        epoch_start_time = time.time()

        if (epoch + 1) % 16 == 0:
            ae_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        i = 0
        for x in batches:
            i=i+1
            ae.train()
            ae.zero_grad()
            rec = ae(x)

            loss_re = loss_fn (rec, x)
            (loss_re).backward()
            ae_optimizer.step()
            rec_loss += loss_re.item()



            #############################################
            os.makedirs('results_ori', exist_ok=True)
            os.makedirs('results_rec', exist_ok=True)


            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time


            if epoch%20==0 and i == 5:
            #if epoch  == 0 and i==1:
                rec_loss /= i
                #kl_loss /= i
                print('\n[%d/%d] - ptime: %.2f, rec loss: %.9f' % (
                    (epoch + 1), train_epoch, per_epoch_ptime, rec_loss))
                rec_loss = 0
                with torch.no_grad():
                    ae.eval()
                    x_rec= ae(x)

                    x=x.cpu()

                    x_rec=x_rec.cpu()


                    # save_image(resultsample.view(-1, 3, im_size, im_size),
                    #            'results_rec/sample_' + str(epoch) + "_" + str(i) + '.png')
                    for j in range(20,29):
                        org_img = transforms.ToPILImage()(x[j].squeeze(0)).convert('L')
                        rec_img = transforms.ToPILImage()(x_rec[j].squeeze(0)).convert('L')

                        org_img.save('results_ori/ori_' + str(epoch) + "_" + str(i) +"_"+str(j)+ '.png')
                        rec_img.save('results_rec/rec_' + str(epoch) + "_" + str(i) + "_"+str(j)+ '.png')


                    # resultsample = x_rec * 0.5 + 0.5
                    # resultsample = resultsample.cpu()
                    # save_image(resultsample.view(-1, 3, im_size, im_size),
                    #            'results_gen/sample_' + str(epoch) + "_" + str(i) + '.png')


        del batches
        del data_train
    print("Training finish!... save training results")
    # output_latent_space = open('./latent_space.pkl', 'wb')
    # # output_bce = open('./BCE_loss.pkl', 'wb')
    # pickle.dump(noise_list, output_latent_space)
    # # pickle.dump(BCE_list, output_bce)
    # output_latent_space.close()
    # # output_bce.close()
    torch.save(ae.state_dict(), "AEmodel.pkl")


if __name__ == '__main__':
    main()
