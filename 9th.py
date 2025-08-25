import os
import numpy as np

from torchvision.utils import save_image

import torch.nn as nn
import torch
from mods import *
from scipy import stats
import copy
import math

ldim = 18
n_imgs = 784 #2**ldim
n_imgs2 = 784
img_size = 1*28*28
img_shape = (1, 28, 28)

    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.appr1 = Hidden_BlocksS(3, img_size, n_imgs2, img_size)
        self.appr2 = Hidden_BlocksS(3, img_size, n_imgs2, img_size)
        self.appr3 = Hidden_BlocksS(3, img_size, n_imgs2, img_size)
        

    def train(self, img):       
        img = self.appr1.train(img)
        img = self.appr2.train(img)
        img = self.appr3.train(img)
        return img
    
    def forward(self, img, nsteps):
        for i in range(nsteps):
            img = self.appr1(img)
            img = self.appr2(img)
            img = self.appr3(img)

        return img
    

d = "../datasets/hf64_rgb.npy"
train_imgs2 = get_digits(80)
train_imgs2 = torch.tensor(train_imgs2).float().reshape(800, img_size)
train_imgs2 = scale_array(train_imgs2, 0,1,-1,1)
#inv_train_imgs2 = scale(1-scale(train_imgs2), out_range=(-1,1))

train_imgs = train_imgs2[:n_imgs]
#inv_train_imgs = inv_train_imgs2#[:n_imgs]
test_imgs = train_imgs2[n_imgs:]

'''
n = 3
ti = train_imgs[:n].reshape(n, 1, 28,28).numpy()
img = switch_tiles(ti, (14,14))
img = torch.tensor(img).float()
save_image(img.reshape(n, *img_shape), "images/pimg.png", nrow=5, normalize=True)
'''
'''
n = 3
noise = torch.tensor(np.random.uniform(-1,1,(64, 784))).float()
img = get_data_kind(noise, train_imgs)
#img = torch.tensor(img).float()
save_image(img.reshape(64, *img_shape), "images/nimg.png", nrow=5, normalize=True)
'''

#print(test_imgs.shape)
save_image(train_imgs[:64].reshape(train_imgs[:64].shape[0], *img_shape), "images/train_imgs.png", nrow=5, normalize=True)

save_image(test_imgs.reshape(test_imgs.shape[0], *img_shape), "images/test_imgs.png", nrow=5, normalize=True)

# Loss function
loss_func = torch.nn.MSELoss()

# Initialize generator
generator =  Generator() #torch.load("models/9th_generator") #

# Optimizers
optim_g = torch.optim.Adam(generator.parameters(), lr = 0.001)

# ----------
#  Training
# ----------

def get_in_datal(bs):
    noise = torch.tensor(np.random.uniform(-1,1,(bs, 784))).float()
    out_imgs = []
    for i in range(bs):
        indcs = np.random.randint(0, n_imgs, n_imgs).tolist()
        imgs = train_imgs[indcs]
        img1 = torch.diag(imgs).unsqueeze(0)
        in_imgs.append(img1)
    
    return torch.cat(in_imgs)


bs = 64

def get_noise(bs):
    noises = []
    imgs = []
    for i in range(bs):
        noise = torch.tensor(np.random.uniform(-1,1,(1, 784))).float()
        diffs = torch.abs(train_imgs - noise)
        #print(diffs)
        #print(diffs.shape)
        losses = torch.sum(diffs, dim =1)
        idx = torch.argmin(losses)
        #print(losses)
        #print(losses.shape); quit()
        noises.append(noise)
        imgs.append(train_imgs[idx].unsqueeze(0))
    noises = torch.cat(noises)
    imgs = torch.cat(imgs)
    return noises, imgs

for epoch in range(100000000):
    indcs = np.random.randint(0, n_imgs, bs).tolist()
    
    noise = torch.tensor(np.random.uniform(0,1,(64, 784))).float()
    #print(noise.shape); quit()
    #a = 1#torch.tensor(np.random.uniform(0, 1, (64,1))).float()
    in_imgs = noise #a*noise + (1-a)*train_imgs[indcs] #get_in_data(train_imgs, bs)


    #print(out_imgs.shape)
    #print(noise.shape)
    out_imgs = compare_images(in_imgs, train_imgs)
    #print(out_imgs.shape)
    #save_image(out_imgs[:64].reshape(64, *img_shape), "images/out_imgs.png", nrow=5, normalize=True)
    #quit()

    optim_g.zero_grad()
    rtimgs1 = generator.appr1(in_imgs)
    r_loss1 = loss_func(rtimgs1, out_imgs)
    r_loss1.backward()
    optim_g.step()

    out_imgs2 = compare_images(rtimgs1.detach(), train_imgs)

    optim_g.zero_grad()
    rtimgs2 = generator.appr2(rtimgs1.detach())
    r_loss2 = loss_func(rtimgs2, out_imgs2)
    r_loss2.backward()
    optim_g.step()

    out_imgs3 = compare_images(rtimgs2.detach(), train_imgs)

    optim_g.zero_grad()
    rtimgs3 = generator.appr3(rtimgs2.detach())
    r_loss3 = loss_func(rtimgs3, out_imgs3)
    r_loss3.backward()
    optim_g.step()

    optim_g.zero_grad()
    rtimgs = generator.train(train_imgs[indcs])
    r_loss = loss_func(rtimgs, train_imgs[indcs])
    r_loss.backward()
    optim_g.step()

    if epoch % 1000 == 0:
        rtimgs = torch.cat([rtimgs1, rtimgs2, rtimgs3]) #rtimgs2
        out_imgs = torch.cat([out_imgs, out_imgs2, out_imgs3])
        noisy_imgs = torch.cat([in_imgs])
        losses = [r_loss1.item(), r_loss2.item(), r_loss3.item(), r_loss.item()] # r_loss2.item()

        Printer(f"{epoch = },  {losses = }") #{g_loss.item() = },

        save_image(out_imgs.reshape(out_imgs.shape[0], *img_shape), "images/out_imgs.png", nrow=5, normalize=True)
        save_image(rtimgs.reshape(rtimgs.shape[0], *img_shape), "images/rtimgs.png", nrow=5, normalize=True)
        save_image(noisy_imgs.reshape(noisy_imgs.shape[0], *(1,28,28)), "images/noisy_imgs.png", nrow=5, normalize=True)

        noise_zs = torch.tensor(np.random.uniform(0, 1, (64,784))).float()
        recr_noise_imgs = generator(noise_zs, 1)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/step1_g_noise_imgs.png", nrow=5, normalize=True)

        noise_zs = torch.tensor(np.random.uniform(0, 1, (64,784))).float()
        recr_noise_imgs = generator(noise_zs, 9)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/step9_g_noise_imgs.png", nrow=5, normalize=True)
        
        noise_zs = switch_tiles(out_imgs.reshape(out_imgs.shape[0],1,28,28),(4,4))[:64]
        noise_zs = torch.tensor(noise_zs).float().reshape(64,784)
        recr_noise_imgs = generator(noise_zs, 1)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/step20_g_noise_imgs.png", nrow=5, normalize=True)

        recr_noise_imgs = generator(test_imgs, 1)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/rtest_imgs.png", nrow=5, normalize=True)

        recr_noise_imgs = generator(test_imgs, 10)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/step10_rtest_imgs.png", nrow=5, normalize=True)

        recr_noise_imgs = generator(train_imgs[indcs], 1)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/n_rtimgs.png", nrow=5, normalize=True)

        recr_noise_imgs = generator(out_imgs, 1)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/r_out_imgs.png", nrow=5, normalize=True)

        i1 = np.random.randint(0, n_imgs, 2).tolist()
        l = 4
        z1 = train_imgs[i1[0]]
        z2 = train_imgs[i1[1]]
        zs = []
        for i in range(l+1):
            a = (i/l)
            z = (1-a)*z1 + a*z2
            zs.append(z.numpy())
        zs = torch.tensor(np.array(zs)).float()
        bimg = generator(zs, 1)
        save_image(bimg.reshape(5, *img_shape), f"images/bimg.png", nrow=5, normalize=True)

        #quit()
        torch.save(generator, "models/9th_generator")

        
        

