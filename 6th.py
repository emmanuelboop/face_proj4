import os
import numpy as np

from torchvision.utils import save_image

import torch.nn as nn
import torch
from mods import *
from scipy import stats
import copy
import math

ldim = 10
n_imgs = 10
img_size = 1*28*28
img_shape = (1, 28, 28)

simg_size = 1*1
simg_shape = (1, 1, 1)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.input_layer =  nn.Sequential(
            #Hidden_Blocks(2, ldim),

            #nn.LayerNorm(ldim),
            nn.Linear(ldim, 640, bias = True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.middle_layer = nn.Sequential(
            Hidden_Blocks(2, 640, nl = True)
        )

        self.output_layer = nn.Sequential(
            nn.LayerNorm(640),
            nn.Linear(640, img_size, bias = True),
            nn.Tanh()
        )

    def forward(self, x):
        #print(x)
        img = self.input_layer(x)
        #print(img)
        #print(img.shape)
        #print(img[0][0].item())
        #print(img[1][0].item())
        
        img = self.middle_layer(img)
        #print(img); quit()
        img = self.output_layer(img)
        return img

d = "../datasets/hf64_rgb.npy"
train_imgs = get_digits(1)
train_imgs = torch.tensor(train_imgs).float().reshape(10, img_size)
save_image(train_imgs.reshape(train_imgs.shape[0], *img_shape), "images/train_imgs.png", nrow=5, normalize=True)
print(train_imgs)

ld_train_imgs = get_digits_wr(1, size = simg_shape[1:])
print(ld_train_imgs.shape)
ld_train_imgs = torch.tensor(ld_train_imgs).float()
save_image(ld_train_imgs.reshape(ld_train_imgs.shape[0], *simg_shape), "images/ld_train_imgs.png", nrow=5, normalize=True)
ld_train_imgs = ld_train_imgs.reshape(10, simg_size)


# Loss function
loss_func = torch.nn.MSELoss()

# Initialize generator
generator =  Generator() #torch.load("models/6th_generator") #

# Optimizers
optim_g = torch.optim.Adam(generator.parameters(), lr = 0.001)

# ----------
#  Training
# ----------

dc_ld = DataContainer(ld_train_imgs)

bs = 10

for epoch in range(100000000):

    ldimgs, indcs = get_inputzs10(ldim)

    optim_g.zero_grad()
    gen_imgs = generator(ldimgs)
    g_loss = loss_func(gen_imgs, train_imgs[indcs])
    g_loss.backward()
    optim_g.step()
    
    
    if epoch % 1000 == 0:
        Printer(f"{epoch = }, {g_loss.item() = }, ") #
        
        save_image(gen_imgs.reshape(gen_imgs.shape[0], *img_shape), "images/gen_imgs.png", nrow=5, normalize=True)
        save_image(ldimgs.reshape(ldimgs.shape[0], *(1,10,1)), "images/ldimgs.png", nrow=5, normalize=True)

        noise_zs = torch.tensor(np.random.uniform(0, 1, (25,10))).float()
        recr_noise_imgs = generator(noise_zs)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/g_recr_noise_imgs.png", nrow=5, normalize=True)

        z = (np.array(list(range(10)))+1)/10
        z = z.tolist()
        my_shuffle(z)
        #[2] = 0.3
        #z[8] = 0.9
        #z[5] = 0.6
        #print(z)
        
        
        noise_zs = torch.tensor([z]).float()
        recr_noise_imgs = generator(noise_zs)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/avei.png", nrow=5, normalize=True)

        noise_zs = expand_zs(ld_train_imgs)
        recr_noise_imgs = generator(noise_zs)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/dupimgs.png", nrow=5, normalize=True)

        l = 10
        z1 = ldimgs[0] 
        z2 = ldimgs[1] 
        zs = []
        for i in range(l+1):
            a = (i/l)
            z = (1-a)*z1 + a*z2
            zs.append(z.numpy())
        zs = torch.tensor(zs).float()
        bimg = generator(zs)
        save_image(bimg.reshape(11, *img_shape), f"images/bimg.png", nrow=5, normalize=True)

        torch.save(generator, "models/6th_generator")

