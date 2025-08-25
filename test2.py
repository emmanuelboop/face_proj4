import os
import numpy as np

from torchvision.utils import save_image

import torch.nn as nn
import torch
from mods import *
from scipy import stats
import copy
import math

n_imgs = 784
img_size = 1*28*28
img_shape = (1, 28, 28)

d = "../datasets/hf64_rgb.npy"

'''
train_imgs2 = get_digits(80)
train_imgs2 = torch.tensor(train_imgs2).float().reshape(800, img_size)
train_imgs2 = scale_array(train_imgs2, 0 ,1,-1,1)
'''

train_imgs2 = torch.tensor(convert_imgs_to_gs(get_faces_wr(0, 800, size = (28,28)))).float()
#print(train_imgs2)
#print(train_imgs2.shape)
#quit()

train_imgs = train_imgs2[:n_imgs]

#name = "cartoon_faces/personai_icartoonface_rectrain_00074/personai_icartoonface_rectrain_00074_0000056.jpg"
#noise = jpg_to_grayscale_resize_and_array(name, (28,28))#get_in_data(train_imgs.reshape(784, 784), 1) #torch.tensor(np.random.uniform(0,1,(1, 784))).float()

cimg1 = train_imgs[77].reshape(784)
cimg2 = train_imgs[365].reshape(784)
cis = np.random.randint(0,n_imgs, np.random.randint(0,n_imgs)).tolist()
noise = copy.deepcopy(cimg1)
noise[cis] = cimg2[cis]

#noise = 0.5*cimg1 + 0.5*cimg2 #torch.tensor(noise).float()/256
#print(noise)
#print(noise.shape)
#print(torch.max(noise))
#quit()
#noise_img = get_data_kind(noise, train_imgs)

save_image(cimg1.reshape(1, *img_shape), "images/cimg1.png", nrow=5, normalize=True)
save_image(cimg2.reshape(1, *img_shape), "images/cimg2.png", nrow=5, normalize=True)

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

def combine_tiles_with_sum(tiles, input_shape, stride):
    height, width = input_shape
    v_stride, h_stride = stride
    output_array = np.zeros(input_shape, dtype=tiles[0].dtype)
    tile_index = 0

    for i in range(0, height - tiles[0].shape[0] + 1, v_stride):
        for j in range(0, width - tiles[0].shape[1] + 1, h_stride):
            output_array[i:i + tiles[0].shape[0], j:j + tiles[0].shape[1]] += tiles[tile_index]
            tile_index += 1

    return output_array

def combine_tiles(tiles, input_shape, stride):
    height, width = input_shape
    v_stride, h_stride = stride
    output_array = np.zeros(input_shape, dtype=tiles[0].dtype)

    tile_index = 0
    for i in range(0, height - tiles[0].shape[0] + 1, v_stride):
        for j in range(0, width - tiles[0].shape[1] + 1, h_stride):
            output_array[i:i + tiles[0].shape[0], j:j + tiles[0].shape[1]] = tiles[tile_index]
            tile_index += 1

    return output_array

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

def combine_tiles_with_brightest(tiles, input_shape, stride):
    height, width = input_shape
    v_stride, h_stride = stride
    output_array = np.zeros(input_shape, dtype=tiles[0].dtype)

    for i in range(0, height - tiles[0].shape[0] + 1, v_stride):
        for j in range(0, width - tiles[0].shape[1] + 1, h_stride):
            tile = tiles.pop(0)
            output_array[i:i + tile.shape[0], j:j + tile.shape[1]] = np.maximum(
                output_array[i:i + tile.shape[0], j:j + tile.shape[1]], tile
            )

    return output_array

def combine_tiles_with_random(tiles, input_shape, stride):
    height, width = input_shape
    v_stride, h_stride = stride
    output_array = np.full(input_shape, np.nan, dtype=tiles[0].dtype)

    for i in range(0, height - tiles[0].shape[0] + 1, v_stride):
        for j in range(0, width - tiles[0].shape[1] + 1, h_stride):
            tile = tiles.pop(0)
            mask = np.isnan(output_array[i:i + tile.shape[0], j:j + tile.shape[1]])
            valid_pixels = tile[~np.isnan(tile)]
            if len(valid_pixels) > 0:
                random_indices = np.random.choice(len(valid_pixels), size=np.sum(mask), replace=True)
                random_values = valid_pixels[random_indices]
                output_array[i:i + tile.shape[0], j:j + tile.shape[1]][mask] = random_values

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
        
    #print(new_tiles[0])
    new_noise_data = combine_tiles_with_average(new_tiles, (28,28), stride)
    #new_noise_data = new_noise_data/np.max(new_noise_data)
    #print(new_noise_data[0])
    #quit()
    #new_noise_data = scale_array(new_noise_data, 0, 1)
    new_noise_data = torch.tensor(new_noise_data).float()
    #print("scaled")
    return new_noise_data

init_noise_img = noise.reshape(28,28)
save_image(init_noise_img.reshape(1, *img_shape), "images/noise_img.png", nrow=5, normalize=True)
train_imgs = train_imgs.reshape(784, 28, 28)

#in_noise_img = init_noise_img
for i in range(1, 29):
    #in_noise_img = init_noise_img
    indcs1 = np.random.randint(0, n_imgs, 1).tolist()
    indcs2 = np.random.randint(0, n_imgs, 1).tolist()

    imgs1 = train_imgs[indcs1].reshape(784)
    imgs2 = train_imgs[indcs2].reshape(784)
    cis = np.random.choice(784, 392, replace=False).tolist()

    in_noise_img = copy.deepcopy(imgs1)
    in_noise_img[cis] = imgs2[cis] 
    in_noise_img = in_noise_img.reshape(28,28)

    #cis = np.random.randint(0, n_imgs, 784//2).tolist()
    #in_noise_img = in_noise_img.reshape(784)
    #ci = np.random.randint(0,n_imgs)
    #in_noise_img[cis] = train_imgs[ci].reshape(784)[cis]
    #in_noise_img = in_noise_img.reshape(28,28)
    imgs = torch.cat([imgs1, imgs2])

    save_image(in_noise_img.reshape(1, *img_shape), f"images/emerged_imgs/in_noise_img{i}.png", nrow=5, normalize=True)
    save_image(imgs.reshape(2, *img_shape), f"images/emerged_imgs/imgs{i}.png", nrow=5, normalize=True)
    #save_image(imgs2.reshape(1, *img_shape), f"images/emerged_imgs/imgs2{i}.png", nrow=5, normalize=True)
    
    imgs = imgs.reshape(2,28,28)

    out_noise_img = img_emerge(in_noise_img, imgs, kernel_size = (7,7), stride = (1,1))
    save_image(out_noise_img.reshape(1, *img_shape), f"images/emerged_imgs/emerged_img{i}.png", nrow=5, normalize=True)
    '''
    if i in [1,2,4,7,14]:
        in_noise_img = img_emerge(out_noise_img, train_imgs, kernel_size = (i,i), stride = (i,i))
        save_image(in_noise_img.reshape(1, *img_shape), f"images/emerged_imgs/crisp_emerged_img{i}.png", nrow=5, normalize=True) 
        #(init_noise_img + out_noise_img + in_noise_img)/3
    '''

    print(i)












