import os
import numpy as np

from torchvision.utils import save_image

import torch.nn as nn
import torch
from mods import *
from scipy import stats
import copy
import math
from mods2 import *


n_imgs = 800
c_imgs = n_imgs - 64
img_size = 3*128*128
img_shape = (3, 128, 128)

d = "../datasets/hf64_rgb.npy"

'''
train_imgs2 = get_digits(80)
train_imgs2 = torch.tensor(train_imgs2).float().reshape(800, img_size)
train_imgs2 = scale_array(train_imgs2, 0 ,1,-1,1)
'''

train_imgs2 = torch.tensor(get_faces_wr(0, n_imgs, size = (128,128))).float()

train_imgs = copy.deepcopy(train_imgs2[:c_imgs])

#name = "cartoon_faces/personai_icartoonface_rectrain_00074/personai_icartoonface_rectrain_00074_0000056.jpg"
#noise = jpg_to_grayscale_resize_and_array(name, (128,128))/255#get_in_data(train_imgs.reshape(784, 784), 1) #torch.tensor(np.random.uniform(0,1,(1, 784))).float()
#noise = torch.tensor(noise.reshape(1,1,128,128)).float()

query_image =  copy.deepcopy(train_imgs2[-2]).unsqueeze(0)
save_image(query_image.reshape(1, *img_shape), f"images/emerged_imgs/query_image.png", nrow=5, normalize=True)

a = 4
edit_area = np.array([85,100,50,80])

#condition = query_image[:,:,edit_area[0]:edit_area[1],edit_area[2]:edit_area[3]]
#query_image[~condition] =  0#torch.tensor(np.random.uniform(0,1,(edit_height,edit_width))).float()

for i in range(1, 10):
    new_query_image, new_dataset, gen_img = get_new_image(query_image, train_imgs, edit_area, a=a, kernel_size = (9,9), stride = (1,1))

    save_image(new_dataset.reshape(new_dataset.shape[0], *img_shape), f"images/emerged_imgs/new_dataset{i}.png", nrow=5, normalize=True)
    save_image(new_query_image.reshape(1, *img_shape), f"images/emerged_imgs/new_query_image{i}.png", nrow=5, normalize=True)
    save_image(gen_img.reshape(1, *img_shape), f"images/emerged_imgs/gen_img{i}.png", nrow=5, normalize=True)

    #print(gen_img.shape)
    #query_image[:,:,edit_area[0]:edit_area[1],edit_area[2]:edit_area[3]] = gen_img.unsqueeze(0)[:,:,edit_area[0]:edit_area[1],edit_area[2]:edit_area[3]]
    #save_image(query_image.reshape(1, *img_shape), f"images/emerged_imgs/old_query_image{i}.png", nrow=5, normalize=True)

    query_image = copy.deepcopy(gen_img.reshape(1,3,128,128))
    
    print(i)













