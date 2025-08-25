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
        self.appr4 = Hidden_BlocksS(3, img_size, n_imgs2, img_size)
        self.appr5 = Hidden_BlocksS(3, img_size, n_imgs2, img_size)
        self.appr6 = Hidden_BlocksS(3, img_size, n_imgs2, img_size)
        self.appr7 = Hidden_BlocksS(3, img_size, n_imgs2, img_size)
        self.appr8 = Hidden_BlocksS(3, img_size, n_imgs2, img_size)
        self.appr9 = Hidden_BlocksS(3, img_size, n_imgs2, img_size)
        
    def appr1_pass(self, img, nsteps):       
        for i in range(nsteps):
            img = self.appr1(img)
        
        return img
    

    def forward(self, img, nsteps):
        for i in range(nsteps):
            img = self.appr1(img)
            img = self.appr2(img)
            img = self.appr3(img)
            img = self.appr4(img)
            img = self.appr5(img)
            img = self.appr6(img)
            img = self.appr7(img)
            img = self.appr8(img)
            img = self.appr9(img)

        return img
    

d = "../datasets/hf64_rgb.npy"
train_imgs2 = get_digits(80)
train_imgs2 = torch.tensor(train_imgs2).float().reshape(800, img_size)
#train_imgs2 = scale_array(train_imgs2, 0,1,-1,1)

train_imgs = train_imgs2[:n_imgs]
test_imgs = train_imgs2[n_imgs:]


#print(test_imgs.shape)
save_image(train_imgs[:64].reshape(train_imgs[:64].shape[0], *img_shape), "images/train_imgs.png", nrow=5, normalize=True)

save_image(test_imgs.reshape(test_imgs.shape[0], *img_shape), "images/test_imgs.png", nrow=5, normalize=True)

# Loss function
loss_func = torch.nn.MSELoss()

# Initialize generator
generator = Generator() # torch.load("models/1st_generator")  #

# Optimizers
optim_g = torch.optim.Adam(generator.parameters(), lr = 0.001)

# ----------
#  Training
# ----------

bs = 64

rs_train_imgs = train_imgs.reshape(n_imgs, 1, 28, 28)


for epoch in range(12000):
    indcs = np.random.randint(0, n_imgs, bs).tolist()

    noise = torch.tensor(np.random.uniform(-1, 1, (64, 784))).float()

    a = torch.tensor(np.random.uniform(0,0.5,(64,1))).float()
    in_imgs = a*noise + (1-a)*train_imgs[indcs]
    out_imgs = train_imgs[indcs]
    
    optim_g.zero_grad()
    rtimgs1 = generator.appr1(in_imgs)
    r_loss1 = loss_func(rtimgs1, out_imgs)
    r_loss1.backward()
    optim_g.step()

    optim_g.zero_grad()
    rtimgs2 = generator.appr2(in_imgs)
    r_loss2 = loss_func(rtimgs2, out_imgs)
    r_loss2.backward()
    optim_g.step()

    optim_g.zero_grad()
    rtimgs3 = generator.appr3(in_imgs)
    r_loss3 = loss_func(rtimgs3, out_imgs)
    r_loss3.backward()
    optim_g.step()

    optim_g.zero_grad()
    rtimgs4 = generator.appr4(in_imgs)
    r_loss4 = loss_func(rtimgs4, out_imgs)
    r_loss4.backward()
    optim_g.step()

    optim_g.zero_grad()
    rtimgs5 = generator.appr5(in_imgs)
    r_loss5 = loss_func(rtimgs5, out_imgs)
    r_loss5.backward()
    optim_g.step()

    optim_g.zero_grad()
    rtimgs6 = generator.appr6(in_imgs)
    r_loss6 = loss_func(rtimgs6, out_imgs)
    r_loss6.backward()
    optim_g.step()

    optim_g.zero_grad()
    rtimgs7 = generator.appr7(in_imgs)
    r_loss7 = loss_func(rtimgs7, out_imgs)
    r_loss7.backward()
    optim_g.step()

    optim_g.zero_grad()
    rtimgs8 = generator.appr8(in_imgs)
    r_loss8 = loss_func(rtimgs8, out_imgs)
    r_loss8.backward()
    optim_g.step()

    optim_g.zero_grad()
    rtimgs9 = generator.appr9(in_imgs)
    r_loss9 = loss_func(rtimgs9, out_imgs)
    r_loss9.backward()
    optim_g.step()
    
    #optim_g.zero_grad()
    #rtimgs_f = generator(out_imgs, 1)
    #r_lossf = loss_func(rtimgs_f, out_imgs)
    #r_lossf.backward()
    #optim_g.step()
    

    if epoch % 1000 == 0:
        rtimgs = torch.cat([rtimgs1, rtimgs2, rtimgs3, rtimgs4]) 
        out_imgs = torch.cat([out_imgs])
        in_imgs = torch.cat([in_imgs , rtimgs5, rtimgs6, rtimgs7])
        losses = [r_loss1.item(), r_loss2.item(), r_loss3.item(), r_loss4.item()] 

        Printer(f"{epoch = },  {losses = }") #{g_loss.item() = },

        save_image(out_imgs.reshape(out_imgs.shape[0], *img_shape), "images/out_imgs.png", nrow=5, normalize=True)
        save_image(rtimgs.reshape(rtimgs.shape[0], *img_shape), "images/rtimgs.png", nrow=5, normalize=True)
        save_image(in_imgs.reshape(in_imgs.shape[0], *(1,28,28)), "images/in_imgs.png", nrow=5, normalize=True)

        #indcs = np.random.randint(0, n_imgs, bs).tolist()
        cpixels = np.random.choice(784, 600, replace = False).tolist()
        ein_imgs = train_imgs[indcs]
        ein_imgs[:, cpixels] = -1
        recr_noise_imgs = generator(ein_imgs, 1)
        save_image(ein_imgs.reshape(ein_imgs.shape[0], *(1,28,28)), "images/ein_imgs.png", nrow=5, normalize=True)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/broken_g_noise_imgs.png", nrow=5, normalize=True)

        noise_zs = torch.tensor(np.random.uniform(-1, 1, (64,784))).float()
        recr_noise_imgs = generator(noise_zs, 1)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/step1_g_noise_imgs.png", nrow=5, normalize=True)

        #noise_zs = torch.tensor(np.random.uniform(-1, 1, (64,784))).float()
        recr_noise_imgs = generator(noise_zs, 3)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/step15_g_noise_imgs.png", nrow=5, normalize=True)
        
        recr_noise_imgs = generator(noise_zs, 20)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/step20_g_noise_imgs.png", nrow=5, normalize=True)

        recr_noise_imgs = generator(test_imgs, 1)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/rtest_imgs.png", nrow=5, normalize=True)

        otimgs = generator.appr1(noise_zs)
        recr_noise_imgs = compare_images(otimgs, train_imgs)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/cmp_imgs.png", nrow=5, normalize=True)

        recr_noise_imgs = generator(0.1*train_imgs[indcs] + 0.9*noise_zs, 1)
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
        torch.save(generator, "models/1st_generator")

        
        

