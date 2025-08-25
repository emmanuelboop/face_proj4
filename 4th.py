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
        
    def appr1_pass(self, img, nsteps):       
        for i in range(nsteps):
            img = self.appr1(img)
            img = self.appr2(img)
            img = self.appr3(img)
        
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
#train_imgs2 = scale_array(train_imgs2, 0,1,-1,1)

train_imgs = train_imgs2[:n_imgs]
test_imgs = train_imgs2[n_imgs:]


#print(test_imgs.shape)
save_image(train_imgs[:64].reshape(train_imgs[:64].shape[0], *img_shape), "images/train_imgs.png", nrow=5, normalize=True)

save_image(test_imgs.reshape(test_imgs.shape[0], *img_shape), "images/test_imgs.png", nrow=5, normalize=True)

# Loss function
loss_func = torch.nn.MSELoss()

# Initialize generator
generator =  Generator() #torch.load("models/4th_generator")  #

# Optimizers
optim_g = torch.optim.Adam(generator.parameters(), lr = 0.001)

# ----------
#  Training
# ----------

bs = 64

rs_train_imgs = train_imgs.reshape(n_imgs, 1, 28, 28)

def find_most_similar(images1, images2, images3):
    # Calculate the Mean Squared Error (MSE) for each pair of images along the second dimension (axis 1)
    mse12 = torch.mean((images1 - images2) ** 2, dim=1)
    mse13 = torch.mean((images1 - images3) ** 2, dim=1)

    # Determine which images are more similar based on MSE
    most_similar = torch.stack([mse12, mse13], dim = 1)
    most_similar = torch.min(most_similar, dim=1).indices

    st_images = torch.stack([images2, images3], dim = 1)

    indcs = list(range(images1.shape[0]))
    most_similar_images = st_images[indcs, most_similar.tolist()]

    return most_similar_images


for epoch in range(100000000):
    indcs1 = np.random.randint(0, n_imgs, bs).tolist()
    indcs2 = np.random.randint(0, n_imgs, bs).tolist()

    imgs1 = train_imgs[indcs1]
    imgs2 = train_imgs[indcs2]
    cis = np.random.choice(784, 392, replace = False).tolist()

    mimgs = copy.deepcopy(imgs1)
    mimgs[:,cis] = imgs2[:,cis]

    simgs1 = find_most_similar(mimgs, imgs1, imgs2)

    imgs3 = copy.deepcopy(imgs1)
    cis2 = np.random.choice(784, 100, replace = False).tolist()
    imgs3[:,cis2] = imgs2[:, cis2]

    #noise = torch.tensor(np.random.uniform(-1, 1, (64, 784))).float()
    #in_imgs1 = 0.7*noise + 0.3*train_imgs[indcs1]
    #simgs1 = compare_images(mimgs, train_imgs)
    mimgs = torch.cat([mimgs, imgs3])
    simgs1 = torch.cat([simgs1, imgs3])

    optim_g.zero_grad()
    rtimgs1 = generator.appr1(mimgs)
    r_loss1 = loss_func(rtimgs1, simgs1)
    r_loss1.backward()
    optim_g.step()

    
    simgs2 = find_most_similar(rtimgs1.detach()[:64], imgs1, imgs2)
    #rtimgs1 = torch.cat([rtimgs1.detach()[:64], imgs3])
    simgs2 = torch.cat([simgs2, imgs3])

    optim_g.zero_grad()
    rtimgs2 = generator.appr2(rtimgs1.detach())
    r_loss2 = loss_func(rtimgs2, simgs2)
    r_loss2.backward()
    optim_g.step()

    
    simgs3 = find_most_similar(rtimgs2.detach()[:64], imgs1, imgs2)
    #rtimgs2 = torch.cat([rtimgs2.detach(), imgs3])
    simgs3 = torch.cat([simgs3, imgs3])

    optim_g.zero_grad()
    rtimgs3 = generator.appr3(rtimgs2.detach())
    r_loss3 = loss_func(rtimgs3, simgs3)
    r_loss3.backward()
    optim_g.step()
    

    if epoch % 1000 == 0:
        rtimgs = torch.cat([rtimgs1, rtimgs2, rtimgs3]) 
        c_imgs = torch.cat([imgs1, imgs2])
        m_imgs = torch.cat([mimgs])
        losses = [r_loss1.item(), r_loss2.item(), r_loss3.item()] 

        Printer(f"{epoch = },  {losses = }") #{g_loss.item() = },

        save_image(simgs1.reshape(simgs1.shape[0], *img_shape), "images/simgs1.png", nrow=5, normalize=True)
        save_image(rtimgs1.reshape(rtimgs1.shape[0], *img_shape), "images/rtimgs1.png", nrow=5, normalize=True)
        save_image(imgs1.reshape(imgs1.shape[0], *img_shape), "images/imgs1.png", nrow=5, normalize=True)
        save_image(imgs2.reshape(imgs2.shape[0], *img_shape), "images/imgs2.png", nrow=5, normalize=True)
        save_image(rtimgs3.reshape(rtimgs3.shape[0], *img_shape), "images/rtimgs3.png", nrow=5, normalize=True)

        save_image(c_imgs.reshape(c_imgs.shape[0], *img_shape), "images/c_imgs.png", nrow=5, normalize=True)
        save_image(rtimgs.reshape(rtimgs.shape[0], *img_shape), "images/rtimgs.png", nrow=5, normalize=True)
        save_image(m_imgs.reshape(m_imgs.shape[0], *(1,28,28)), "images/m_imgs.png", nrow=5, normalize=True)

        #indcs = np.random.randint(0, n_imgs, bs).tolist()
        cpixels = np.random.choice(784, 600, replace = False).tolist()
        ein_imgs = train_imgs[indcs1]
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

        recr_noise_imgs = generator(simgs1, 1)
        save_image(recr_noise_imgs[:64].reshape(recr_noise_imgs[:64].shape[0], *img_shape), "images/n_rtimgs.png", nrow=5, normalize=True)

        

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
        torch.save(generator, "models/4th_generator")

        
        

