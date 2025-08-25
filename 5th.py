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
        #self.appr2 = Hidden_BlocksS(3, img_size, n_imgs2, img_size)
        #self.appr3 = Hidden_BlocksS(3, img_size, n_imgs2, img_size)
        
    def appr1_pass(self, img, nsteps):       
        for i in range(nsteps):
            img = self.appr1(img)
            #img = self.appr2(img)
            #img = self.appr3(img)
        
        return img
    

    def forward(self, img, nsteps):
        for i in range(nsteps):
            img = self.appr1(img)
            #img = self.appr2(img)
            #img = self.appr3(img)

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
generator =  Generator() #torch.load("models/5th_generator")  #

# Optimizers
optim_g = torch.optim.Adam(generator.parameters(), lr = 0.001)

# ----------
#  Training
# ----------

bs = 64

rs_train_imgs = train_imgs.reshape(n_imgs, 1, 28, 28)

def split_into_tiles(arr, kernel_size, stride):
    k_height, k_width = kernel_size
    v_stride, h_stride = stride

    height, width = arr.shape

    # Calculate the number of tiles in each dimension
    num_tiles_vert = (height - k_height) // v_stride + 1
    num_tiles_horiz = (width - k_width) // h_stride + 1

    # Calculate the shape of the output 4D array
    new_shape = (
        num_tiles_vert,
        num_tiles_horiz,
        k_height,
        k_width,
    )

    # Use NumPy strides to create the tiles without explicit loops
    tile_view = np.lib.stride_tricks.as_strided(
        arr,
        shape=new_shape,
        strides=(
            arr.strides[0] * v_stride,
            arr.strides[1] * h_stride,
            arr.strides[0],
            arr.strides[1],
        ),
    )

    return torch.tensor(tile_view.reshape(-1, k_height, k_width)).float()

def combine_tiles_with_average(tiles, input_shape, stride):
    height, width = input_shape
    v_stride, h_stride = stride
    output_array = np.zeros(input_shape, dtype=tiles[0].dtype)
    tile_index = 0

    for i in range(0, height - tiles[0].shape[0] + 1, v_stride):
        for j in range(0, width - tiles[0].shape[1] + 1, h_stride):
            tile = tiles[tile_index]
            output_array[i:i + tile.shape[0], j:j + tile.shape[1]] += tile
            tile_index += 1

    # Count how many times each position was added (used for averaging)
    count_array = np.zeros(input_shape, dtype=int)
    tile_index = 0

    for i in range(0, height - tiles[0].shape[0] + 1, v_stride):
        for j in range(0, width - tiles[0].shape[1] + 1, h_stride):
            count_array[i:i + tiles[0].shape[0], j:j + tiles[0].shape[1]] += 1
            tile_index += 1

    # Calculate the average by dividing by the count (avoiding division by zero)
    count_array[count_array == 0] = 1  # Avoid division by zero
    output_array /= count_array

    return output_array

def img_emerge(noise_data, train_data, kernel_size = (2,2), stride = (1,1)):
    splitted_noise_data = split_into_tiles(noise_data.numpy(), kernel_size, stride)

    new_tiles = []
    for i in range(len(splitted_noise_data)):
        losses = []
        stds = []
        for j in range(train_data.shape[0]):
            snd = splitted_noise_data[i]
            std = split_into_tiles(train_data[j].numpy(), kernel_size, stride)[i]
            stds.append(std)

            diff = torch.abs(snd - std)
            loss = torch.sum(diff)
            losses.append(loss.item())
        
        losses = np.array(losses)
        min_index = np.argmin(losses)
        new_tiles.append(stds[min_index].numpy())
        
    new_noise_data = combine_tiles_with_average(new_tiles, (28,28), stride)
    new_noise_data = torch.tensor(new_noise_data).float()
    return new_noise_data

def find_most_similar(images1, images2, images3, kernel_size = (7,7), stride = (1,1)):
    images1 = images1.reshape(64, 28, 28)
    images2 = images2.reshape(64, 28, 28)
    images3 = images3.reshape(64, 28, 28)

    gimgs = []
    for img1, img2, img3 in zip(images1, images2, images3):
        simg1 = split_into_tiles(img1.numpy(), kernel_size, stride)
        simg2 = split_into_tiles(img2.numpy(), kernel_size, stride)
        simg3 = split_into_tiles(img3.numpy(), kernel_size, stride)

        simg1 = simg1.reshape(simg1.shape[0], simg1.shape[1]*simg1.shape[2])
        simg2 = simg2.reshape(simg2.shape[0], simg2.shape[1]*simg2.shape[2])
        simg3 = simg3.reshape(simg3.shape[0], simg3.shape[1]*simg3.shape[2])

        l1 = torch.sum(torch.abs(simg1 - simg2), dim =1)
        l2 = torch.sum(torch.abs(simg1 - simg3), dim =1)

        most_similar = torch.stack([l1, l2], dim = 1)
        most_similar = torch.min(most_similar, dim=1).indices

        st_images = torch.stack([simg2, simg3], dim = 1)

        indcs = list(range(simg1.shape[0]))
        msi = st_images[indcs, most_similar.tolist()]

        msi = msi.reshape(msi.shape[0],*kernel_size)
        
        new_data = combine_tiles_with_average(msi.numpy(), (28, 28), stride)
        new_data = torch.tensor(new_data).float()
        new_data = new_data.reshape(784).unsqueeze(0)
        gimgs.append(new_data)
        


    gimgs = torch.cat(gimgs)
    #print(gimgs.shape); quit()
    return gimgs

for epoch in range(100000000):
    indcs1 = np.random.randint(0, n_imgs, bs).tolist()
    indcs2 = np.random.randint(0, n_imgs, bs).tolist()

    imgs1 = train_imgs[indcs1]
    imgs2 = train_imgs[indcs2]
    cis = np.random.choice(784, 392, replace = False).tolist()

    mimgs = copy.deepcopy(imgs1)
    mimgs[:,cis] = imgs2[:,cis]

    simgs1 = find_most_similar(mimgs, imgs1, imgs2)

    optim_g.zero_grad()
    rtimgs1 = generator.appr1(mimgs)
    r_loss1 = loss_func(rtimgs1, simgs1)
    r_loss1.backward()
    optim_g.step()
    

    if epoch % 1000 == 0:
        rtimgs = torch.cat([rtimgs1]) 
        c_imgs = torch.cat([imgs1, imgs2])
        m_imgs = torch.cat([mimgs])
        losses = [r_loss1.item()] 

        Printer(f"{epoch = },  {losses = }") #{g_loss.item() = },

        save_image(simgs1.reshape(simgs1.shape[0], *img_shape), "images/simgs1.png", nrow=5, normalize=True)
        save_image(rtimgs1.reshape(rtimgs1.shape[0], *img_shape), "images/rtimgs1.png", nrow=5, normalize=True)
        save_image(imgs1.reshape(imgs1.shape[0], *img_shape), "images/imgs1.png", nrow=5, normalize=True)
        save_image(imgs2.reshape(imgs2.shape[0], *img_shape), "images/imgs2.png", nrow=5, normalize=True)
        #save_image(rtimgs3.reshape(rtimgs3.shape[0], *img_shape), "images/rtimgs3.png", nrow=5, normalize=True)

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
        torch.save(generator, "models/5th_generator")

        
        

