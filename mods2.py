import os
import numpy as np

from torchvision.utils import save_image

import torch.nn as nn
import torch
from mods import *
from scipy import stats
import copy
import math


def combine_tiles_with_average(tiles, input_shape, stride):
    height, width = input_shape
    v_stride, h_stride = stride
    output_array = np.zeros(input_shape, dtype=tiles[0].dtype)
    tile_index = 0

    for j in range(0, width - tiles[0].shape[1] + 1, h_stride):
        for i in range(0, height - tiles[0].shape[0] + 1, v_stride):
            tile = tiles[tile_index]
            output_array[i:i + tile.shape[0], j:j + tile.shape[1]] += tile
            tile_index += 1

    # Count how many times each position was added (used for averaging)
    count_array = np.zeros(input_shape, dtype=int)
    tile_index = 0

    for j in range(0, width - tiles[0].shape[1] + 1, h_stride):
        for i in range(0, height - tiles[0].shape[0] + 1, v_stride):
        
            count_array[i:i + tiles[0].shape[0], j:j + tiles[0].shape[1]] += 1
            tile_index += 1

    # Calculate the average by dividing by the count (avoiding division by zero)
    count_array[count_array == 0] = 1  # Avoid division by zero
    output_array /= count_array

    return output_array

def combine_tiles_with_average3d(tiles, input_shape, stride):
    channels, height, width = input_shape
    v_stride, h_stride = stride
    output_array = np.zeros(input_shape, dtype=tiles[0].dtype)
    tile_index = 0

    for j in range(0, width - tiles[0].shape[2] + 1, h_stride):
        for i in range(0, height - tiles[0].shape[1] + 1, v_stride):
            tile = tiles[tile_index]
            output_array[:, i:i + tile.shape[1], j:j + tile.shape[2]] += tile
            tile_index += 1

    # Count how many times each position was added (used for averaging)
    count_array = np.zeros(input_shape, dtype=int)
    tile_index = 0

    for j in range(0, width - tiles[0].shape[2] + 1, h_stride):
        for i in range(0, height - tiles[0].shape[1] + 1, v_stride):
            count_array[:, i:i + tiles[0].shape[1], j:j + tiles[0].shape[2]] += 1
            tile_index += 1

    # Calculate the average by dividing by the count (avoiding division by zero)
    count_array[count_array == 0] = 1  # Avoid division by zero
    output_array /= count_array

    return output_array

def img_emerge_waqi(query_image, reference_images, kernel_size = (2,2), stride = (1,1)):

    '''
    Note: kernel_size and stride is in the form (h, w)
    '''

    query_image_height = query_image.shape[2]
    query_image_width = query_image.shape[3]

    generated_tiles = []
    for r in range(0, query_image_width, stride[0]):
        for c in range(0, query_image_height, stride[1]):
            qi_selected_tile = query_image[:,:,c:c+kernel_size[0],r:r+kernel_size[1]]
            ri_selected_tiles = reference_images[:,:,c:c+kernel_size[0],r:r+kernel_size[1]]

            qi_selected_tile_f = qi_selected_tile.reshape(query_image.shape[0],query_image.shape[1]*kernel_size[0]*kernel_size[1])
            ri_selected_tiles_f = ri_selected_tiles.reshape(reference_images.shape[0],reference_images.shape[1]*kernel_size[0]*kernel_size[1])

            loss = torch.sum(torch.abs(ri_selected_tiles_f - qi_selected_tile_f), dim = 1)
            ci = torch.min(loss, dim=0).indices

            generated_tiles.append(ri_selected_tiles[ci].numpy()[0])
            query_image[:,:,c:c+kernel_size[0],r:r+kernel_size[1]] = (query_image[:,:,c:c+kernel_size[0],r:r+kernel_size[1]] + ri_selected_tiles[ci])/2
            
            if c+kernel_size[0] == query_image_width:
                break

        if r+kernel_size[1] == query_image_height:
                break
    
    #print(generated_tiles[0].shape)
    generated_image = combine_tiles_with_average(generated_tiles, (query_image_height, query_image_width), stride)
    generated_image = torch.tensor(generated_image).float()
    return generated_image

def img_emerge_wqi(query_image, reference_images, kernel_size = (2,2), stride = (1,1)):

    '''
    Note: kernel_size and stride is in the form (h, w)
    '''

    query_image_height = query_image.shape[2]
    query_image_width = query_image.shape[3]

    for r in range(0, query_image_width, stride[0]):
        for c in range(0, query_image_height, stride[1]):
            qi_selected_tile = query_image[:,:,c:c+kernel_size[0],r:r+kernel_size[1]]
            ri_selected_tiles = reference_images[:,:,c:c+kernel_size[0],r:r+kernel_size[1]]

            qi_selected_tile_f = qi_selected_tile.reshape(query_image.shape[0],query_image.shape[1]*kernel_size[0]*kernel_size[1])
            ri_selected_tiles_f = ri_selected_tiles.reshape(reference_images.shape[0],reference_images.shape[1]*kernel_size[0]*kernel_size[1])

            loss = torch.sum(torch.abs(ri_selected_tiles_f - qi_selected_tile_f), dim = 1)
            ci = torch.min(loss, dim=0).indices

            query_image[:,:,c:c+kernel_size[0],r:r+kernel_size[1]] = (query_image[:,:,c:c+kernel_size[0],r:r+kernel_size[1]] + ri_selected_tiles[ci])/2
            
            if c+kernel_size[0] == query_image_width:
                break
        
        if r+kernel_size[1] == query_image_height:
                break
    
    return query_image

def img_emerge_wa(query_image, reference_images, kernel_size = (2,2), stride = (1,1)):

    '''
    Note: kernel_size and stride is in the form (h, w)
    '''

    out_image = copy.deepcopy(query_image)*0
    query_image_height = query_image.shape[2]
    query_image_width = query_image.shape[3]

    #generated_tiles = []
    for r in range(0, query_image_width, stride[0]):
        for c in range(0, query_image_height, stride[1]):
            qi_selected_tile = query_image[:,:,c:c+kernel_size[0],r:r+kernel_size[1]]
            ri_selected_tiles = reference_images[:,:,c:c+kernel_size[0],r:r+kernel_size[1]]

            qi_selected_tile_f = qi_selected_tile.reshape(query_image.shape[0],query_image.shape[1]*kernel_size[0]*kernel_size[1])
            ri_selected_tiles_f = ri_selected_tiles.reshape(reference_images.shape[0],reference_images.shape[1]*kernel_size[0]*kernel_size[1])

            loss = torch.sum(torch.abs(ri_selected_tiles_f - qi_selected_tile_f), dim = 1)
            ci = torch.min(loss, dim=0).indices
            
            out_image[:,:,c:c+kernel_size[0],r:r+kernel_size[1]] = (out_image[:,:,c:c+kernel_size[0],r:r+kernel_size[1]] + ri_selected_tiles[ci])/2

            #generated_tiles.append(ri_selected_tiles[ci].numpy()[0])
            
            if c+kernel_size[0] == query_image_width:
                break

        if r+kernel_size[1] == query_image_height:
                break
    
    #print(generated_tiles[0].shape)
    #generated_image = combine_tiles_with_average(generated_tiles, (query_image_height, query_image_width), stride)
    #generated_image = torch.tensor(generated_image).float()
    return out_image #generated_image

def img_emerge_wa2(query_image, reference_images, kernel_size = (2,2), stride = (1,1)):

    '''
    Note: kernel_size and stride is in the form (h, w)
    '''

    query_image_height = query_image.shape[2]
    query_image_width = query_image.shape[3]

    generated_tiles = []
    for r in range(0, query_image_width, stride[0]):
        for c in range(0, query_image_height, stride[1]):
            qi_selected_tile = query_image[:,:,c:c+kernel_size[0],r:r+kernel_size[1]]
            ri_selected_tiles = reference_images[:,:,c:c+kernel_size[0],r:r+kernel_size[1]]

            qi_selected_tile_f = qi_selected_tile.reshape(query_image.shape[0],query_image.shape[1]*kernel_size[0]*kernel_size[1])
            ri_selected_tiles_f = ri_selected_tiles.reshape(reference_images.shape[0],reference_images.shape[1]*kernel_size[0]*kernel_size[1])

            loss = torch.sum(torch.abs(ri_selected_tiles_f - qi_selected_tile_f), dim = 1)
            ci = torch.min(loss, dim=0).indices

            generated_tiles.append(ri_selected_tiles[ci].numpy())
            
            if c+kernel_size[0] == query_image_width:
                break

        if r+kernel_size[1] == query_image_height:
                break
    
    #print(generated_tiles[0].shape)
    generated_image = combine_tiles_with_average3d(generated_tiles, (3, query_image_height, query_image_width), stride)
    generated_image = torch.tensor(generated_image).float()
    return generated_image


def sort_arrays_by_losses(arrays1, losses):
    if len(arrays1) != len(losses):
        raise ValueError("Input arrays must have the same length.")

    # Use torch.argsort to get the indices that would sort losses in ascending order
    sorted_indices = torch.argsort(losses)
    # Use the sorted indices to rearrange arrays1
    sorted_arrays1 = arrays1[sorted_indices]
    
    return sorted_arrays1

def get_new_imagel(query_image, dataset, edit_area, kernel_size = (2,2), stride = (1,1)):
    query_image = copy.deepcopy(query_image)
    dataset = copy.deepcopy(dataset)
    
    losses = torch.sum(torch.abs(dataset.reshape(dataset.shape[0],dataset.shape[2]*dataset.shape[3]) - query_image.reshape(1,query_image.shape[2]*query_image.shape[3])), dim=1)
    sorted_dataset = sort_arrays_by_losses(dataset, losses)

    chosen_dataset = torch.cat([sorted_dataset[:64], copy.deepcopy(query_image)])
    
    cis = np.random.choice(128*128, (128*128)//2, replace=False).tolist()
    query_image = query_image.flatten()
    query_image[cis] = 0
    query_image = query_image.reshape(1,1,128,128)

    gen_img = img_emerge_wa(query_image, chosen_dataset, kernel_size = kernel_size, stride = stride).squeeze(0)
    return query_image, chosen_dataset, gen_img

def modify_region(arr, selected_region, new_value, select_inverse=True):
    """
    Modify the selected region or its inverse of a PyTorch tensor.

    Args:
    - arr (torch.Tensor): The input PyTorch tensor of shape (b, c, h, w).
    - selected_region (list): The selected region specified as [sr, er, sc, ec].
    - new_value: The value to set for the selected region or its inverse.
    - select_inverse (bool, optional): If True, select the inverse of the region; if False, select the region.

    Returns:
    - modified_tensor (torch.Tensor): The modified tensor with the selected region or its inverse changed to new_value.
    """
    b, c, h, w = arr.shape
    sr, er, sc, ec = selected_region

    # Create a mask for the selected region or its inverse
    mask = torch.ones((b, c, h, w), dtype=arr.dtype)
    
    if select_inverse:
        mask = torch.ones((b, c, h, w), dtype=arr.dtype)
        mask[:, :, sr:er, sc:ec] = 0
        modified_tensor = arr.clone()
        modified_tensor[mask == 1] = new_value
    else:
        modified_tensor = arr.clone()
        modified_tensor[:, :, sr:er, sc:ec] = new_value
       
    # Set the selected region or its inverse to the new_value
    

    return modified_tensor

def get_sorted_dataset(data, dataset, edit_area, a = 7):
    new_data = copy.deepcopy(data)
    new_dataset = copy.deepcopy(dataset)
    edit_area = copy.deepcopy(edit_area)

    edit_area[0] -= a
    edit_area[2] -= a
    edit_area[1] += a
    edit_area[3] += a

    new_data = modify_region(new_data, edit_area, 0, select_inverse=True)
    new_data[:,:,edit_area[0]+a:edit_area[1]-a,edit_area[2]+a:edit_area[3]-a] = 0

    new_dataset = modify_region(new_dataset, edit_area, 0, select_inverse=True)
    new_dataset[:,:,edit_area[0]+a:edit_area[1]-a,edit_area[2]+a:edit_area[3]-a] = 0

    losses = torch.sum(torch.abs(new_dataset.reshape(new_dataset.shape[0],new_dataset.shape[1]*new_dataset.shape[2]*new_dataset.shape[3]) - new_data.reshape(1,new_data.shape[1]*new_data.shape[2]*new_data.shape[3])), dim=1)
    sorted_dataset = sort_arrays_by_losses(dataset, losses)
    return sorted_dataset


def get_new_image(query_image, dataset, edit_area, a = 7, kernel_size = (2,2), stride = (1,1)):
    #query_image = copy.deepcopy(query_image)
    #dataset = copy.deepcopy(dataset)
    sorted_dataset = get_sorted_dataset(query_image, dataset, edit_area, a=a)

    query_image[:,:,edit_area[0]:edit_area[1],edit_area[2]:edit_area[3]] = 0
    #cs = 64
    #cimg = torch.sum(sorted_dataset[:cs], dim = 0)/cs
    #cimg = cimg.unsqueeze(0)
    #print(cimg.shape)
    chosen_dataset = torch.cat([sorted_dataset[2].unsqueeze(0), copy.deepcopy(query_image)])

    edit_area[0] -= a
    edit_area[2] -= a
    edit_area[1] += a
    edit_area[3] += a

    j = copy.deepcopy(query_image[:,:,edit_area[0]-a:edit_area[1]+a,edit_area[2]-a:edit_area[3]+a])
    k = copy.deepcopy(j)
    j[k < 0.5] = 1
    j[k > 0.5] = 0
    query_image[:,:,edit_area[0]-a:edit_area[1]+a,edit_area[2]-a:edit_area[3]+a] = j

    gen_img = img_emerge_wa2(query_image, chosen_dataset, kernel_size = kernel_size, stride = stride).squeeze(0)
    return query_image, chosen_dataset, gen_img
