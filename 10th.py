import os
import numpy as np

from torchvision.utils import save_image

import torch.nn as nn
import torch
from mods import *
from scipy import stats
import copy
import math

ldim = 9
n_imgs = 512 #2**ldim
img_size = 1*28*28
img_shape = (1, 28, 28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.encoder_layers = nn.Sequential(
            nn.Linear(img_size, n_imgs, bias = True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.LayerNorm(n_imgs),
            nn.Linear(n_imgs, n_imgs, bias = True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.LayerNorm(n_imgs),
            nn.Linear(n_imgs, ldim, bias = True),
            #nn.Tanh()
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.bnb1 = nn.Sequential(
            nn.Linear(ldim, ldim, bias = True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.LayerNorm(ldim),
            nn.Linear(ldim, ldim, bias = True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.LayerNorm(ldim),
            nn.Linear(ldim, ldim, bias = True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.LayerNorm(ldim),
            nn.Linear(ldim, ldim, bias = True),
            nn.Tanh()
        )

        self.bnb2 = nn.Sequential(
            nn.Linear(ldim, ldim, bias = True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.LayerNorm(ldim),
            nn.Linear(ldim, ldim, bias = True),
            nn.Tanh()
        )

        self.decoder_layers = nn.Sequential(
            nn.Linear(ldim, n_imgs, bias = True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.LayerNorm(n_imgs),
            nn.Linear(n_imgs, n_imgs, bias = True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.LayerNorm(n_imgs),
            nn.Linear(n_imgs, n_imgs, bias = True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.LayerNorm(n_imgs),
            nn.Linear(n_imgs, img_size, bias = True),
            nn.Tanh()
        )

    def leaky_scalar(self, x):
        mask_low = x <= 0.5
        mask_high = x > 0.5

        new_x = x.clone()  # Create a new tensor with the same values as x
        new_x[mask_low] /= 1.1
        new_x[mask_high] *= 1.1

        return new_x
    
    def reconstruct(self, img):
        z = self.encoder_layers(img)

        z = self.bnb1(z)
        z = self.bnb2(z)

        rimg = self.decoder_layers(z)
        return rimg

    def forward(self, x):
        x = self.bnb1(x)
        x = self.bnb2(x)
        img = self.decoder_layers(x)
        return img
    

d = "../datasets/hf64_rgb.npy"
train_imgs2 = get_digits(56)
train_imgs2 = torch.tensor(train_imgs2).float().reshape(560, img_size)

train_imgs = train_imgs2[:n_imgs]
test_imgs = train_imgs2[n_imgs:]

save_image(train_imgs[:64].reshape(train_imgs[:64].shape[0], *img_shape), "images/train_imgs.png", nrow=5, normalize=True)
save_image(test_imgs.reshape(test_imgs.shape[0], *img_shape), "images/test_imgs.png", nrow=5, normalize=True)

# Loss function
loss_func = torch.nn.MSELoss()

# Initialize generator
generator =  Generator() #torch.load("models/10th_generator") #

# Optimizers
optim_g = torch.optim.Adam(generator.parameters(), lr = 0.001)
optim_e = torch.optim.Adam(generator.encoder_layers.parameters(), lr = 0.001)
optim_d = torch.optim.Adam(generator.decoder_layers.parameters(), lr = 0.001)

# ----------
#  Training
# ----------

#zinputs = get_combinations(ldim)
#zinputs = torch.tensor(zinputs).float()

#zinputs[zinputs < 1] = -1

#lzinputs = zinputs.tolist()

def get_indcs(o_z):
    indcs = []
    for n_z in o_z.tolist():
        indices = lzinputs.index(n_z)
        indcs.append(indices)

    return indcs

bs = 64

def get_izs(ldim):
    zs = torch.tensor(np.random.uniform(-1,1,(bs, ldim))).float()
    #zs = change_values_in_range(zs, 0.45, 0.5, 0.45)
    #zs = change_values_in_range(zs, 0.5, 0.55, 0.55)
    #print(arr)
    return zs

for epoch in range(100000000):
    ldimgs = get_izs(ldim)

    dzs = copy.deepcopy(ldimgs)
    dzs[dzs > 0] = 1
    dzs[dzs <= 0] = -1

    optim_g.zero_grad()
    dldimgs = generator.bnb1(ldimgs)
    l_loss = loss_func(dldimgs, dzs)
    l_loss.backward()
    optim_g.step()

    dzs = copy.deepcopy(dldimgs.detach())
    dzs[dzs > 0] = 1
    dzs[dzs <= 0] = -1

    optim_g.zero_grad()
    dldimgs = generator.bnb2(dldimgs.detach())
    l_loss2 = loss_func(dldimgs, dzs)
    l_loss2.backward()
    optim_g.step()

    
    indcs = np.random.randint(0,n_imgs,bs).tolist()

    in_imgs = 0.5*train_imgs[indcs] + 0.5*torch.tensor(np.random.uniform(-1, 1, (64,784))).float()
    optim_g.zero_grad()
    eimgs = generator.encoder_layers(in_imgs)
    dldimgs = generator.bnb1(eimgs)
    dldimgs = generator.bnb2(dldimgs)
    rtimgs = generator.decoder_layers(dldimgs)
    r_loss = loss_func(rtimgs, train_imgs[indcs])
    r_loss.backward()
    optim_d.step()
    optim_e.step()
    

    rtimgs = torch.cat([in_imgs, rtimgs])
    losses = [l_loss.item(), l_loss2.item(), r_loss.item()]

    if epoch % 1000 == 0:
        Printer(f"{epoch = }, {losses = }  ") #
        save_image(rtimgs.reshape(rtimgs.shape[0], *img_shape), "images/rtimgs.png", nrow=5, normalize=True)
        save_image(train_imgs[indcs].reshape(train_imgs[indcs].shape[0], *img_shape), "images/train_imgs_indcs.png", nrow=5, normalize=True)

        
        noise_zs = torch.tensor(np.random.uniform(-1, 1, (64,784))).float() 
        recr_noise_imgs = generator.reconstruct(noise_zs)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/isgnoise_imgs.png", nrow=5, normalize=True)

        noise_zs = torch.tensor(np.random.uniform(-1, 1, (64,ldim))).float() 
        recr_noise_imgs = generator(noise_zs)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/lsnoise_imgs.png", nrow=5, normalize=True)
        
        recr_test_imgs = generator.reconstruct(test_imgs)
        save_image(recr_test_imgs[:64].reshape(recr_test_imgs[:64].shape[0], *img_shape), "images/recr_test_imgs.png", nrow=5, normalize=True)

        recr_train_imgs = generator.reconstruct(train_imgs[indcs])
        save_image(recr_train_imgs[:64].reshape(recr_train_imgs[:64].shape[0], *img_shape), "images/recr_train_imgs.png", nrow=5, normalize=True)

        ns = torch.tensor(np.random.uniform(-1, 1, (64,784))).float() 
        l = 4
        z1 = ns[0] 
        z2 = ns[8] 
        zs = []
        for i in range(l+1):
            a = (i/l)
            z = (1-a)*z1 + a*z2
            zs.append(z.numpy())
        zs = torch.tensor(np.array(zs)).float()
        bimg = generator.reconstruct(zs)
        save_image(bimg.reshape(5, *img_shape), f"images/bimg.png", nrow=5, normalize=True)

        #quit()
        
        torch.save(generator, "models/10th_generator")
        

