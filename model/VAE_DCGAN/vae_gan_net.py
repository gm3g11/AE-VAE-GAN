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

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels=1, hiden_size=128, output_channels=64):
        super(Encoder, self).__init__()
        # input parameters
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.features = nn.Sequential(
            # 1 x 128 x 128
            nn.Conv2d(self.input_channels, output_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            # 64 x 64 x 64
            nn.Conv2d(output_channels, output_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(output_channels * 2),
            nn.ReLU(),
            # 128 x 32 x 32
            nn.Conv2d(output_channels * 2, output_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(output_channels * 4),
            nn.ReLU(),
            # 256 x 16 x 16
            nn.Conv2d(output_channels * 4, output_channels * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(output_channels * 8),
            nn.ReLU(),
            # 512 x 8 x 8
            nn.Conv2d(output_channels * 8, output_channels * 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(output_channels * 16),
            nn.ReLU())
            # 1024 x 4 x 4

        self.mean = nn.Sequential(
            nn.Linear(output_channels * 16 * 4 * 4, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, hiden_size))

        self.logvar = nn.Sequential(
            nn.Linear(output_channels * 16 * 4 * 4, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, hiden_size))

    def forward(self, x):
        batch_size = x.size()[0]

        hidden_representation = self.features(x)

        mean = self.mean(hidden_representation.view(batch_size, -1))
        logvar = self.logvar(hidden_representation.view(batch_size, -1))

        return mean, logvar

    def hidden_layer(self, x):
        batch_size = x.size()[0]
        output = self.features(x)
        return output


class Decoder(nn.Module):
    def __init__(self, input_size, representation_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.representation_size = representation_size
        dim = representation_size[0] * representation_size[1] * representation_size[2]

        self.preprocess = nn.Sequential(
            nn.Linear(input_size, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU())


        # 1024x4x4
        self.deconv0 = nn.ConvTranspose2d(representation_size[0], 512, 4, stride=2, padding=1)
        self.act0 = nn.Sequential(nn.BatchNorm2d(512),
                                  nn.ReLU())

        # 512 x 8 x 8
        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.act1 = nn.Sequential(nn.BatchNorm2d(256),
                                  nn.ReLU())
        # 256 x 16 x 16
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.act2 = nn.Sequential(nn.BatchNorm2d(128),
                                  nn.ReLU())
        # 128 x 32 x 32
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.act3 = nn.Sequential(nn.BatchNorm2d(64),
                                  nn.ReLU())
        # 64 x 64 x 64
        self.deconv4 = nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1)
        # 1 x 128 x 128
        #self.activation = nn.Sigmoid()

    def forward(self, code):
        bs = code.size()[0]
        preprocessed_codes = self.preprocess(code)
        preprocessed_codes = preprocessed_codes.view(-1,
                                                     self.representation_size[0],
                                                     self.representation_size[1],
                                                     self.representation_size[2])

        output = self.deconv0(preprocessed_codes, output_size=(bs, 512, 8, 8))
        output = self.act0(output)
        output = self.deconv1(output, output_size=(bs, 256, 16, 16))
        output = self.act1(output)
        output = self.deconv2(output, output_size=(bs, 128, 32, 32))
        output = self.act2(output)
        output = self.deconv3(output, output_size=(bs, 64, 64, 64))
        output = self.act3(output)
        output = self.deconv4(output, output_size=(bs, 1, 128, 128))
        output = torch.sigmoid(output)


        # output = self.deconv1(preprocessed_codes, output_size=(bs, 256, 16, 16))
        #
        # output = self.act1(output)
        # output = self.deconv2(output, output_size=(bs, 128, 32, 32))
        # output = self.act2(output)
        # output = self.deconv3(output, output_size=(bs, 64, 64, 64))
        # output = self.act3(output)
        # output = self.deconv4(output, output_size=(bs, 1, 128, 128))
        # output = self.activation(output)
        return output


class VAE_GAN_Generator(nn.Module):
    def __init__(self, input_channels=1, hidden_size=128, representation_size=[1024,4,4],output_channels=64):
        super(VAE_GAN_Generator, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.representation_size = representation_size

        self.encoder = Encoder()
        self.decoder = Decoder(hidden_size, representation_size)

    def forward(self, x):
        batch_size = x.size()[0]
        mean, logvar = self.encoder(x)
        std = logvar.mul(0.5).exp_()

        reparametrized_noise = Variable(torch.randn((batch_size, self.hidden_size))).cuda()
        #reparametrized_noise = Variable(torch.randn((batch_size, self.hidden_size)))
        reparametrized_noise = mean + std * reparametrized_noise

        rec_images = self.decoder(reparametrized_noise)

        return mean, logvar, rec_images


class Discriminator(nn.Module):
    def __init__(self, input_channels=1, representation_size=(1024, 4, 4)):
        super(Discriminator, self).__init__()
        self.representation_size = representation_size
        dim = representation_size[0] * representation_size[1] * representation_size[2]

        self.main = nn.Sequential(
            # 1 x 128 x 128
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
        )

        self.lth_features = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.LeakyReLU(0.2))

        self.sigmoid_output = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid())

    def forward(self, x):
        batch_size = x.size()[0]
        features = self.main(x)
        lth_rep = self.lth_features(features.view(batch_size, -1))
        output = self.sigmoid_output(lth_rep)
        return output

    def similarity(self, x):
        batch_size = x.size()[0]
        features = self.main(x)
        lth_rep = self.lth_features(features.view(batch_size, -1))
        return lth_rep


if __name__ == '__main__':
    x=torch.rand(64*1*128*128)
    x=x.view(64,1,128,128)
    enc=Encoder(1,1024,64)
    mu,var=enc(x)
    print(mu.shape,var.shape)
    gan=VAE_GAN_Generator(input_channels=1, hidden_size=1024, representation_size=[1024,4,4],output_channels=64)
    a,b,c=gan(x)
    dis=Discriminator()
    d=dis.forward(x)
    e=dis.similarity(x)
