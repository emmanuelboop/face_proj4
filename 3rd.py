import os
import numpy as np

from torchvision.utils import save_image

import torch.nn as nn
import torch
from mods import *
from scipy import stats
import copy
import math
#from augment import *

ldim = 64
n_imgs = 60
img_size = 1*28*28
img_shape = (1, 28, 28)

simg_size = 8*8
simg_shape = (1, 8, 8)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.input_layer =  nn.Sequential(
            Hidden_Blocks(5, ldim)
        )

        self.output_layer = nn.Sequential(
            nn.LayerNorm(ldim),
            nn.Linear(ldim, img_size, bias = False),
            nn.Tanh()
        )

    def forward(self, img):
        img = self.input_layer(img)
        img = self.output_layer(img)
        return img

d = "../datasets/hf64_rgb.npy"
train_imgs = get_digits(6)
train_imgs = torch.tensor(train_imgs).float().reshape(n_imgs, img_size)

#train_imgs = (2 * train_imgs) - 1

ld_train_imgs = get_digits_wr(6)
ld_train_imgs = torch.tensor(ld_train_imgs).float().reshape(n_imgs, simg_size)
#ld_train_imgs = (2 * ld_train_imgs) - 1

# Loss function
loss_func = torch.nn.MSELoss()

# Initialize generator
generator =  Generator() #torch.load("models/3rd_generator") #

# Optimizers
optim_g = torch.optim.Adam(generator.parameters(), lr = 0.001)

# ----------
#  Training
# ----------

dc_ld = DataContainer(ld_train_imgs)
dc = DataContainer(train_imgs)

bs = 64

for epoch in range(100000000):  
    ldimgs = dc_ld.get_data(bs)
    timgs = dc.get_data(bs)

    cis = []
    czs = []
    for i in range(60):
        czi = torch.tensor(np.random.normal(0,1,(1,ldim))).float() 
        
        sv = nn.CosineSimilarity()(ld_train_imgs, czi)
        sv2 = torch.sort(sv, descending=True)
        index = sv2.indices[0].item()

        if index not in cis:      
            cis.append(index)
            czs.append(czi[0].numpy())

    czs = torch.tensor(czs).float()
    c_imgs = train_imgs[cis]

    optim_g.zero_grad()
    gen_imgs = generator(czs)
    g_loss = loss_func(gen_imgs, c_imgs)
    g_loss.backward()
    optim_g.step()
    
    if epoch % 50 == 0:
        Printer(f"{epoch = }, {g_loss.item() = }, {len(cis) = }") #
        
        save_image(gen_imgs.reshape(gen_imgs.shape[0], *img_shape), "images/gen_imgs.png", nrow=5, normalize=True)
        save_image(timgs.reshape(timgs.shape[0], *img_shape), "images/timgs.png", nrow=5, normalize=True)
        save_image(c_imgs.reshape(c_imgs.shape[0], *img_shape), "images/c_imgs.png", nrow=5, normalize=True)
        save_image(czs.reshape(czs.shape[0], *simg_shape), "images/czs.png", nrow=5, normalize=True)

        noise_zs = torch.tensor(np.random.normal(0, 1, (25,simg_size))).float()

        recr_noise_imgs = generator(noise_zs)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/g_recr_noise_imgs.png", nrow=5, normalize=True)

        l = 10
        z1 = czs[0] 
        z2 = czs[1] 
        zs = []
        for i in range(l):
            a = (i/l)
            z = (1-a)*z1 + a*z2
            zs.append(z.numpy())
        zs = torch.tensor(zs).float()
        bimg = generator(zs)
        save_image(bimg.reshape(10, *img_shape), f"images/bimg.png", nrow=5, normalize=True)

        torch.save(generator, "models/3rd_generator")

