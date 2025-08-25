from dataclasses import replace
import sys
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import itertools
from torchvision.utils import save_image
import os
import cv2
import csv
import skimage.transform
from skimage.util import invert
from collections import defaultdict
from scipy.stats import rankdata
import copy
import random

def get_crr_imgs(gimgs, span):
	gimgs = gimgs.clone().detach().numpy()
	path = "../face_proj/img_align_celeba_npy"
	file_names = sorted(os.listdir(path))[:span]
	crr_imgs = []
	for gimg in gimgs:
		pv_loss = float('inf')
		crr_img = None
		for file_name in file_names:
			img = np.load(path+"/"+file_name).reshape(12288)
			loss = np.sum(np.abs(img - gimg))
			if loss < pv_loss:
				pv_loss = loss
				crr_img = np.copy(img)
			
		crr_imgs.append(crr_img)
	
	crr_imgs = np.array(crr_imgs)
	return crr_imgs

def get_crr_imgs(gimgs, span):
	gimgs = gimgs.clone().detach().numpy()
	cmp_imgs = np.load("cmp_imgs.npy")[:span]

	crr_imgs = []
	for gimg in gimgs:
		gimg = np.repeat(gimg, span)
		losses = np.sum(np.abs(cmp_imgs - gimg), axis = 0)
		indices = np.argmin(losses)
		crr_imgs.append(cmp_imgs[indices])
				
	crr_imgs = np.array(crr_imgs)
	return crr_imgs

def get_dataset(n_data):
	path = "/home/emmanuel/Downloads/stylegan2-ada-pytorch-main"
	n_imgs = 1024 #len(os.listdir(path+"/images"))
	cis = np.random.choice(n_imgs,n_data)

	imgs = []
	zs = []
	for i in cis:
		img = cv2.imread(path+f"/images/{i}.png")
		try:
			rgb_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		except:
			continue
		r,g,b = cv2.split(rgb_image)
		rgb_image = np.array([r,g,b])/255
		imgs.append(rgb_image)
		
		z = np.load(path+f"/zs/{i}.npy")[0]
		zs.append(z)

	imgs = np.array(imgs)
	zs = np.array(zs)
	return imgs, zs

def create_z(ldim, nzs):
	for i in range(nzs):
		np.save(f"zs/z{i}.npy", np.random.normal(0,1,(ldim)))
	quit()

def get_zs2(start, end):
	path = "org_zs"
	file_names = sorted(os.listdir(path))[start:end]
	zs = []
	for file_name in file_names:
		z = np.load(path+"/"+file_name)
		
		zs.append(z)

	zs = np.array(zs)
	
	return zs

def get_zs(start, end):
	path = "zs"
	file_names = sorted(os.listdir(path))[start:end]
	zs = []
	for file_name in file_names:
		zs.append(np.load(path+"/"+file_name))
	
	zs = np.array(zs)
	return zs

def convert_imgs_to_gs(imgs):
	gs_imgs = []
	for img in imgs:
		img = np.sum(img, axis=0)/3
		gs_imgs.append(img)
	gs_imgs = np.array(gs_imgs)
	return gs_imgs

def get_faces_wr(start, end, size = (16, 16), path = "../face_proj/img_align_celeba_npy"):
	file_names = sorted(os.listdir(path))[start:end]
	imgs = []
	for file_name in file_names:
		img = np.load(path+"/"+file_name)
		img = np.dstack(img)
		img = cv2.resize(img, size)
		r,g,b = cv2.split(img)
		img = np.array([r,g,b])
		imgs.append(img)
	
	imgs = np.array(imgs)
	return imgs

def get_faces(start, end, path = "../face_proj/img_align_celeba_npy"):
	file_names = sorted(os.listdir(path))[start:end]
	imgs = []
	for file_name in file_names:
		img = np.load(path+"/"+file_name)
		imgs.append(img)
	
	imgs = np.array(imgs)
	return imgs

def get_digits(nie, path = "mnist"):
	imgs = []
	for j in range(nie):
		for label in range(10):
			file_name = sorted(os.listdir(path+f"/{label}s"))[j]
			img = np.load(path+f"/{label}s/"+file_name)
			imgs.append(img)
	
	imgs = np.array(imgs)
	return imgs

def get_digits_wr(nie, size = (8, 8), path = "mnist"):
	imgs = []
	for j in range(nie):
		for label in range(10):
			file_name = sorted(os.listdir(path+f"/{label}s"))[j]
			img = np.load(path+f"/{label}s/"+file_name)
			img = cv2.resize(img, size)
			imgs.append(img)
	
	imgs = np.array(imgs)
	return imgs

def my_shuffle(x):
	if len(x) == 1:
		raise Exception
	for i in reversed(range(1, len(x))):
		# pick an element in x[:i] with which to exchange x[i]
		j = int(random.random() * i)
		x[i], x[j] = x[j], x[i]
	
def expand_zsl(ld_train_imgs):
	zs = []
	for i in range(10):
		indcs = list(range(10))
		del indcs[i]
		my_shuffle(indcs)
		indcs.insert(i, i)

		zs.append(ld_train_imgs[indcs].flatten().numpy())

	zs = torch.tensor(zs).float()
	return zs

def expand_zsl4(ld_train_imgs):
	zs = []
	for i in range(10):
		indcs = list(range(10))
		del indcs[i]
		cl = copy.deepcopy(indcs)

		my_shuffle(indcs)
		indcs.insert(i, i)

		for j in range(10):
			cn = np.random.randint(0,10)
			if j != i and j != cn:
				indcs[j] = cn

		zs.append(ld_train_imgs[indcs].flatten().numpy())
	zs = torch.tensor(zs).float()
	return zs


def expand_zsl2(ld_train_imgs):
	zs = []
	for i in range(10):
		indcs = list(range(10))
		del indcs[i]
		my_shuffle(indcs)
		indcs.insert(i, i)
		indcs = (np.array(indcs) + 1)/10
		zs.append(indcs)
	
	zs = torch.tensor(zs).float()
	return zs

def expand_zsl5(ld_train_imgs):
	zs = []
	for i in range(10):
		indcs = np.random.uniform(0,1,10).tolist()
		indcs.sort()
		j = indcs[i]
		del indcs[i]
		my_shuffle(indcs)
		indcs.insert(i, j)
		zs.append(indcs)
	zs = torch.tensor(zs).float()
	return zs

def expand_zsl7(ld_train_imgs):
	zs = []
	for i in range(10):
		indcs = list(range(10))		
		my_shuffle(indcs)

		itcf = list(range(10))
		for j in range(i+1):
			cn = np.random.choice(itcf)
			indcs[cn] = cn
			itcf.remove(cn)

		indcs = (np.array(indcs) + 1)/10

		zs.append(indcs)
	zs = torch.tensor(zs).float()
	return zs

'''
All of the items not in correct positions
'''
def expand_zsl9(ld_train_imgs):
	zs = []
	for i in range(10):
		indcs = list(range(10))		
		my_shuffle(indcs)

		for v in range(10):
			cn = np.random.randint(0,10)
			if v != cn:
				indcs[v] = cn

		indcs = (np.array(indcs) + 1)/10

		zs.append(indcs)
	zs = torch.tensor(zs).float()
	return zs

'''
If n number of items are in the correct positions
with duplicates.
'''
def expand_zsl8(ld_train_imgs):
	zs = []
	for i in range(10):
		indcs = list(range(10))		
		my_shuffle(indcs)

		itcf = list(range(10))
		pcis = []
		for j in range(i+1):
			cn = np.random.choice(itcf)
			indcs[cn] = cn
			pcis.append(cn)
			itcf.remove(cn)

		for v in range(10):
			cn = np.random.randint(0,10)
			if v not in pcis and v != cn:
				indcs[v] = cn

		indcs = (np.array(indcs) + 1)/10

		zs.append(indcs)
	zs = torch.tensor(zs).float()
	return zs

'''
All numbers being duplicated multiple times.
'''
def expand_zsl3(ld_train_imgs):
	zs = []
	for i in range(10):
		indcs = list(range(10))
		del indcs[i]
		
		my_shuffle(indcs)
		indcs.insert(i, i)

		for j in range(10):
			cn = np.random.randint(0,10)
			if j != i and j != cn:
				indcs[j] = cn

		indcs = (np.array(indcs) + 1)/10

		zs.append(indcs)
	zs = torch.tensor(zs).float()
	return zs

def get_inputzs(ldim):
	mz = list(range(ldim))
	zs = []
	indcs = []
	for i in range(ldim):
		while True:
			z = np.random.choice(mz, 10)
			found = False
			for j in range(1, ldim+1):
				if z[-j] == ldim - j:
					z = (z+1)/10
					zs.append(z)
					indcs.append(ldim-j)
					found = True
					break
			
			if found:
				break
			else:
				continue

				
	zs = torch.tensor(zs).float()
	return zs, indcs

def get_inputzs10(ldim):
	nmz = np.array(np.arange(1,ldim+1)).reshape(ldim, 1)
	nmz = torch.tensor(nmz).expand(ldim,ldim)/ldim
	#print(nmz)

	zs = []
	indcs = []
	for i in range(ldim):
		while True:
			z = np.random.uniform(0,1,(1,10))
			#print(z)

			scores = torch.sum(torch.abs(nmz - torch.tensor(z)),1).numpy()
			#print(scores)
			idxs = np.where(scores == scores.min())[0]
			#print(idxs); quit()
			if len(idxs) > 1:
				continue
			else:
				zs.append(z[0])
				indcs.append(idxs[-1])
				break
	
	#print(indcs); quit()
	zs = torch.tensor(zs).float()
	return zs, indcs

def get_inputzs2(ldim):
	mz = list(range(ldim))
	nmz = np.array(mz) + 1
	

	zs = []
	indcs = []
	for i in range(ldim):
		z = np.random.uniform(0,1,10)
		zs.append(z)
		z = np.round(z, 1)*10
		

		diffs = np.abs(nmz - z)
		idxs = np.where(diffs == diffs.min())[0]
		indcs.append(idxs[-1])
	
	#print(indcs); quit()
	zs = torch.tensor(zs).float()
	return zs, indcs

def get_inputzs1(ldim):
	mz = list(range(ldim))
	nmz = np.array(mz)
	
	zs = []
	indcs = []
	for i in range(ldim):
		z = np.random.choice(mz, 10)
		zs.append(z)

		diffs = np.abs(nmz - z)
		idxs = np.where(diffs == diffs.min())[0]
		indcs.append(idxs[-1])
				
	zs = torch.tensor(zs).float()
	return zs, indcs

'''
All numbers being duplicated multiple times with
more than one item being at correct positions.
'''
def expand_zsl10(ld_train_imgs):
	zs = []
	for i in range(10):
		indcs = list(range(10))
		del indcs[i]
		
		my_shuffle(indcs)
		indcs.insert(i, i)

		for j in range(10):
			cn = np.random.randint(0,10)
			if j != i and j != cn:
				indcs[j] = cn
		
		for u in range(i+1):
			iti = np.random.randint(0, i+1)
			indcs[iti] = iti

		indcs = (np.array(indcs) + 1)/10

		zs.append(indcs)
	zs = torch.tensor(zs).float()
	return zs

def expand_zsl6(ld_train_imgs):
	zs = []
	for i in range(10):
		indcs = list(range(10))
		del indcs[i]

		if np.random.randint(0,2) == 1:
			indcs = np.array(indcs)*0
			indcs = indcs.tolist()
			#indcs.insert(i, i)
			indcs.insert(i, np.random.uniform(0,1))

		else:
			my_shuffle(indcs)
			indcs.insert(i, i)

			for j in range(10):
				cn = np.random.randint(0,10)
				if j != i and j != cn:
					indcs[j] = cn

			indcs = (np.array(indcs) + 1)/10

		zs.append(indcs)
	zs = torch.tensor(zs).float()
	return zs

'''
One number being duplicated multiple times.
'''
def expand_zs(ld_train_imgs):
	zs = []
	for i in range(10):
		indcs = list(range(10))
		del indcs[i]
		cl = copy.deepcopy(indcs)

		my_shuffle(indcs)
		indcs.insert(i, i)

		for j in range(np.random.randint(0,10)):
			cn = np.random.randint(0,10)
			indcs[cn] = i
		indcs = (np.array(indcs) + 1)/10
		zs.append(indcs)

	zs = torch.tensor(np.array(zs)).float()
	return zs

def get_attrs(start, end):
	path = "../img_attrs"
	file_names = sorted(os.listdir(path))[start:end]
	img_attrs = []
	for file_name in file_names:
		img_attrs.append(np.load(path+"/"+file_name))
	
	img_attrs = np.array(img_attrs)
	return img_attrs

def get_ffhq(nimgs):
	csv_path = "/home/emmanuel/Downloads/faces_dataset/celeb_a_dataset/list_attr_celeba.csv"
	imgs_path = "/home/emmanuel/Downloads/faces_dataset/celeb_a_dataset/img_align_celeba/img_align_celeba"

	imgs = []
	attrs = []
	with open(csv_path,"r") as lac:
		csv_reader = csv.DictReader(lac)
		field_names = csv_reader.fieldnames 
		
		for img_row in csv_reader:
			img_attrs = []
			for field_name in field_names:

				if img_row[field_name] == "1":
					img_attrs.append(1)

				elif img_row[field_name] == "-1":
					img_attrs.append(0)

				else:
					name = img_row[field_name]
			
			img_attrs = np.array(img_attrs)
			np.save("img_attrs/"+name+".npy", img_attrs)
			#attrs.append(img_attrs)	
	print("done"); quit()
	imgs = np.array(imgs)
	attrs = np.array(attrs)
	return imgs, attrs		

def gfc(category, nimgs):
	category_path = "/home/emmanuel/Downloads/faces_dataset/celeb_a_dataset/img_align_celeba/faces_with_categories/"+category
	img_names = os.listdir(category_path)[:nimgs]
	img_path = "/home/emmanuel/Downloads/faces_dataset/celeb_a_dataset/img_align_celeba/img_align_celeba"

	imgs = []
	for img_name in img_names:
		img = cv2.imread(img_path+"/"+img_name)
		img = cv2.resize(img, (64,64))
		try:
			rgb_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		except:
			continue
		r,g,b = cv2.split(rgb_image)
		rgb_image = np.array([r,g,b])/255
		imgs.append(rgb_image)
		
	imgs = np.array(imgs)
	return imgs

class Printer():
    """Print things to stdout on one line dynamically"""
    def __init__(self,data):
        sys.stdout.write("\r\x1b[K"+data.__str__())
        sys.stdout.flush()

def random_sample(dataset, bs):
	dataset = dataset.clone()
	ci = np.random.choice(len(dataset), bs)
	return dataset[ci]
	
def crop_imgs(imgs, size = (30, 30), offset = 10, fill_value = 1):
	imgs = imgs.clone()
	img_shape = imgs.shape[2:]
	imgs[:,:,offset:offset+size[0],offset:offset+size[1]] = fill_value
	
	return imgs

class DataContainer2:
	def __init__(self, datasets):
		self.datasets = datasets
		self.offset = 0
		
	def get_data(self, bs, random = False):
		if not random:
			cds = []
			for i in range(len(self.datasets)):
				cd = self.datasets[i][self.offset:self.offset+bs]
				cds.append(cd)
				
			if self.offset + bs >= len(self.datasets[0]):
				self.offset = 0
			
			else:
				self.offset += bs
		
		elif random:
			cds = []
			ci = np.random.choice(len(self.dataset), bs)
			for i in range(len(self.datasets)):
				cd = self.datasets[i][ci]
				cds.append(cd)
				
		return cds
		
class DataContainer:
	def __init__(self, dataset):
		self.dataset = dataset
		self.offset = 0
		
	def get_data(self, bs, random = False):
		if not random:
			cd = self.dataset[self.offset:self.offset+bs]
	
			if self.offset + bs >= len(self.dataset):
				self.offset = 0
			
			else:
				self.offset += bs
		
		elif random:
			ci = np.random.choice(len(self.dataset), bs)
			cd = self.dataset[ci]
		
		return copy.deepcopy(cd)

def place_ones(size, count):
	for positions in itertools.combinations(range(size), count):
		p = [0] * size

		for i in positions:
			p[i] = 1

		yield p
      
def get_combinations(z_dim):
	zs = []
	zs.append(torch.zeros((z_dim)).tolist())
	for i in range(1, z_dim+1):
		zs += list(place_ones(z_dim, i))
	return zs

def get_combinations2(z_dim, count):
	zs = list(place_ones(z_dim, count))
	#zs.append(torch.tensor(np.random.normal(0,1,(z_dim))).tolist())
	return torch.tensor(zs).float()
	
def get_more_dataset(dataset, zs_len, mode = 1, nm = 2):
	nri = zs_len - len(dataset)
	adtn_imgs = []
	
	if mode == 1: # generate additional data by blending data in dataset
		for i in range(nri):
			ci = np.random.choice(len(dataset), nm)
			new_img = torch.prod(dataset[ci], 0)
			adtn_imgs.append(new_img.unsqueeze(0))
		
		adtn_imgs = torch.cat(adtn_imgs)
		dataset = torch.cat((dataset, adtn_imgs))
		
	elif mode == 2: # get additional data by duplicating data in dataset
		ci = np.random.choice(len(dataset), nri)
		dataset = torch.cat((dataset, dataset[ci]))
		
	return dataset

class GenDataContainer4:
	def __init__(self, dataset, z_dim, mode = 1, nm = 2):
		self.dataset = dataset
		self.offset = 0
		self.z_dim = z_dim
		
		self.zs_dataset = []
		for i in range(1, len(dataset)+1):
			self.zs_dataset.append(list(place_ones(z_dim, i)))
		
		#print(len(self.zs_dataset))
		if len(self.zs_dataset) > len(self.dataset):
			self.dataset = get_more_dataset(self.dataset, len(self.zs), mode, nm)	
				
		print(f"done preparing data. length of zs: {len(self.zs_dataset)}, length of dataset: {len(self.dataset)}")
			
	def get_data(self, bs):
		cd = self.dataset[self.offset:self.offset+bs]
		
		czs_c = self.zs_dataset[self.offset:self.offset+bs]
		czs = []
		for i in range(len(czs_c)):
			ci = 0#np.random.choice(len(czs_c[i]), 1)
			czs.append(np.array(czs_c[i])[ci])
		czs = torch.tensor(czs).float()
		
		if self.offset + bs >= len(self.dataset):
			self.offset = 0
		
		else:
			self.offset += bs
		
		return czs, cd
		
class GenDataContainer3:
	def __init__(self, dataset, z_dim, mode = 1, nm = 2):
		self.dataset = dataset
		self.offset = 0
		self.z_dim = z_dim
		
		self.zs = torch.tensor(list(place_ones(z_dim, 1))).float() #torch.tensor(list(place_ones(z_dim, z_dim//2))).float()
		print(self.zs)
		if len(self.zs) > len(self.dataset):
			self.dataset = get_more_dataset(self.dataset, len(self.zs), mode, nm)	
				
		print(f"done preparing data. length of zs: {len(self.zs)}, length of dataset: {len(self.dataset)}")
			
	def get_data(self, bs):
		czs = torch.tensor(np.random.normal(0,1,(bs, self.z_dim))).float()
		_, indices = torch.max(czs.clone(), 1)
		cd = self.dataset[indices]
		
		cezs = self.zs[indices]
		czs[:,indices] += np.random.uniform(1,10)
		 
		zs = torch.cat((czs, cezs))
		cd = torch.cat((cd, cd))
		
		return zs, cd
		
		
class GenDataContainer2:
	def __init__(self, dataset, z_dim, mode = 1, nm = 2):
		self.dataset = dataset
		self.offset = 0
		
		self.zs = torch.tensor(list(place_ones(z_dim, 1))).float() #torch.tensor(list(place_ones(z_dim, z_dim//2))).float()
		print(self.zs)
		if len(self.zs) > len(self.dataset):
			self.dataset = get_more_dataset(self.dataset, len(self.zs), mode, nm)	
				
		print(f"done preparing data. length of zs: {len(self.zs)}, length of dataset: {len(self.dataset)}")
			
	def get_data(self, bs, get_imgs = True, get_zs = True):
		if get_zs:
			czs = self.zs[self.offset:self.offset+bs]
			
		if get_imgs:
			cd = self.dataset[self.offset:self.offset+bs]
	
		if self.offset + bs >= len(self.dataset):
			self.offset = 0
		
		else:
			self.offset += bs
		
		if get_imgs and get_zs:
			return czs, cd
		
		elif get_imgs:
			return cd
		
		elif get_zs:
			return czs
			
class GenDataContainer:
	def __init__(self, dataset, z_dim, count, mode = 1, nm = 2):
		self.dataset = dataset
		self.offset = 0
		
		self.zs = get_combinations2(z_dim, count) 
		
		if len(self.zs) > len(self.dataset):
			self.dataset = get_more_dataset(self.dataset, z_dim, mode, nm)	
				
		print(f"done preparing data. length of zs: {len(self.zs)}, length of dataset: {len(self.dataset)}")
			
	def get_data(self, bs, get_imgs = True, get_zs = True):
		if get_zs:
			czs = self.zs[self.offset:self.offset+bs]
			
		if get_imgs:
			cd = self.dataset[self.offset:self.offset+bs]
	
		if self.offset + bs >= len(self.dataset):
			self.offset = 0
		
		else:
			self.offset += bs
		
		if get_imgs and get_zs:
			return czs, cd
		
		elif get_imgs:
			return cd
		
		elif get_zs:
			return czs
			

def normalize(tensor):
	tensor += torch.abs(torch.amin(tensor, 1)).unsqueeze(1)
	tensor = tensor/torch.amax(tensor, 1).unsqueeze(1)
	return tensor

class sigmoid(nn.Module):
	def __init__(self, k = 1):
		super(sigmoid, self).__init__()
		
		self.k = k

	def forward(self, x):
		x = self.k/(1 + 2.71828**(-1*x))
		return x

	def inverse(self, x):
		return torch.log(x/(1 - x))

class LRactivf(nn.Module):
	def __init__(self, in_features):
		super(LRactivf, self).__init__()
		
		self.tparams = torch.tensor(np.random.uniform(0.2, 0.5,(1, in_features))).float()
		self.tparams = nn.Parameter(self.tparams)

	def forward(self, x):
		#print(self.tparams)
		#x[x<0] = x[x<0] * self.tparams[x<0]
		x = torch.where(x < 0, x * self.tparams, x)
		#x = x*self.tparams
		return x



def scale_array(arr, new_min, new_max, old_min=None, old_max=None):
    """
    Scale a numpy array to a new range.

    Parameters:
        arr (numpy.ndarray): The input numpy array to be scaled.
        new_min (float): The minimum value of the new range.
        new_max (float): The maximum value of the new range.
        old_min (float, optional): The minimum value of the old range. Default is None.
        old_max (float, optional): The maximum value of the old range. Default is None.

    Returns:
        numpy.ndarray: The scaled numpy array.
    """
    # If old_min and old_max are not provided, use the minimum and maximum values of the input array
    if old_min is None:
        old_min = np.min(arr)
    if old_max is None:
        old_max = np.max(arr)

    # Scale the array to the new range
    scaled_arr = (arr - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

    return scaled_arr



def scale(x, out_range=(0, 1), axis=None):
	x = x.numpy()
	domain = np.min(x, axis), np.max(x, axis)
	y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
	out = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
	return torch.tensor(out).float()

def normalize_pattern(arrays, method = "ordinal"):
    arrays = rankdata(arrays, method = method, axis = 1) - 1
    return arrays

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>1)

def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

def transfer_pattern(arr1, arr2):
	arr1_pt = rankdata(arr1, method = "ordinal")-1
	arr2 = np.sort(arr2)
	return arr2[arr1_pt]

def transfer_patterns2(arrays1, arrays2):
	out_arrays = []
	for arr1, arr2 in zip(arrays1, arrays2):
		n = 2#np.random.choice([4,8,16,32,64]) #np.random.randint(2,64)
		arr1 = arr1.reshape(64,64)
		arr2 = arr2.reshape(64,64)
		for row in range(0,64,1):
			for col in range(0,64,1):            
				l = col
				k = l+n
				l2 = row
				k2 = l2+n

				if k > 64 or k2 > 64:
					continue

				arr1_blk = arr1[l:k,l2:k2].flatten()
				arr2_blk = arr2[l:k,l2:k2].flatten()

				arr = transfer_pattern(arr1_blk, arr2_blk)

				arr2[l:k,l2:k2] = arr.reshape(k-l,k2-l2)

		out_arrays.append(arr2)
	out_arrays = np.array(out_arrays)
	return out_arrays

def transfer_patterns(arrays1, arrays2):
	out_arrays = []
	for arr1, arr2 in zip(arrays1, arrays2):
		arr = transfer_pattern(arr1, arr2)
		out_arrays.append(arr)
	out_arrays = np.array(out_arrays)
	return out_arrays

class disentangle(nn.Module):
	def __init__(self, ldim):
		super(disentangle, self).__init__()
		self.ldim = ldim - 1

	def forward(self, x):
		codes = normalize_pattern(torch.abs(x.detach()))
		codes = codes * torch.sign(x.detach())
		codes = codes/self.ldim
		x = x + codes		
		return x

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		
		self.middle_linear_network = nn.Sequential(
			nn.LayerNorm(14),
			nn.Linear(14, 2**14),
			nn.Sigmoid(),
			
			nn.LayerNorm(2**14),
			nn.Linear(2**14, 2**14),
			nn.Sigmoid(),
		)
		
	def forward(self, z):
		z = self.middle_linear_network[0](z)
		z = F.linear(z, self.middle_linear_network[1].weight)
		z = self.middle_linear_network[2](z)
		
		z = self.middle_linear_network[3](z)
		z = F.linear(z, self.middle_linear_network[4].weight)
		z = self.middle_linear_network[5](z)
		return z	
	
	
class Mappers(nn.Module):
	def __init__(self, num_of_mappers, in_features):
		super(Mappers, self).__init__()
		
		self.num_of_mappers = num_of_mappers
		self.mappers = nn.ModuleList()
		
		for i in range(num_of_mappers):
			self.mappers.append(

				nn.Sequential(
					nn.LayerNorm(in_features),
					nn.Linear(in_features, in_features, bias = False),
					nn.LeakyReLU(np.random.uniform(0.1,0.9), inplace=True),
				)
			)
		
		self.pmi = 0 # present module index
		
	
	def use_mod(self, x, index):
		z = self.mappers[index](x)
		return z

	def forward(self, x, move):
		if self.pmi == self.num_of_mappers:
			self.pmi = 0
		
		z = self.mappers[self.pmi](x)
		
		if move:
			self.pmi += 1
				
		return z		
	
	
def restructure(vectors, mv):
	svs = nn.CosineSimilarity()(vectors, mv)
	pt = normalize_pattern_1d(svs)
	
	restructured_vectors = []
	for i in range(len(vectors)):
		c = 0
		for e in pt:
			if e == float(i):
				restructured_vectors.append(vectors[c].tolist())
			c += 1

	restructured_vectors = torch.tensor(list(reversed(restructured_vectors))).float()
	
	return restructured_vectors
	
	
class Mappers(nn.Module):
	def __init__(self, in_features, out_features):
		super(Mappers, self).__init__()
		
		self.in_features = in_features
		self.out_features = out_features
		self.mappers = nn.ModuleList()
		
		for i in range(in_features):
			self.mappers.append(

				nn.Sequential(
					nn.Linear(in_features, out_features, bias = False),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Linear(out_features, 1, bias = False),
				)
			)
		
		self.pmi = 0 # present module index
		
	
	def use_mod(self, x, index):
		z = self.mappers[index](x)
		return z

	def forward(self, x, move):
		if self.pmi == self.num_of_mappers:
			self.pmi = 0
		
		z = self.mappers[self.pmi](x)
		
		if move:
			self.pmi += 1
				
		return z			
	
def reweigh(v):
    v = v.clone()
    v1 = torch.rot90(v.expand((4,4)),-1)
    vm = v1*v
    vst1 = nn.Softmax(dim = 1)(vm)
    rv = v*vst1
    vs = torch.sum(rv, 1)
    return vs	

def reconv(v, rbzs):
    vc = normalize_pattern_1d(v)+1

    try:
        i = rbzs.tolist().index(vc.tolist())
    except ValueError:
        i = 0

    return i

class Block(nn.Module):
	def __init__(self, n_nodes):
		super(SCBlock, self).__init__()
		
		self.n_nodes = n_nodes

		#self.nl = nn.LayerNorm(n_nodes)
		self.ll = nn.Linear(n_nodes,n_nodes)
		self.af = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x):
		#x = self.nl(x)
		x = self.ll(x)
		z = self.af(x)
		return z	

class SCBlock(nn.Module):
	def __init__(self, n_nodes):
		super(SCBlock, self).__init__()
		
		self.n_nodes = n_nodes

		self.nl = nn.LayerNorm(n_nodes)
		self.ll = nn.Linear(n_nodes,n_nodes)
		self.af = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x):
		z = self.nl(x)
		z = self.ll(z)
		z = self.af(z)

		z = z + x
		return z			


class SCHidden_Blocks(nn.Module):
	def __init__(self, n_blocks, n_nodes):
		super(SCHidden_Blocks, self).__init__()
		
		self.n_blocks = n_blocks
		self.n_nodes = n_nodes
		self.blocks = nn.ModuleList()

		for i in range(n_blocks):
			layers = [nn.LayerNorm(n_nodes), nn.Linear(n_nodes,n_nodes), nn.LeakyReLU(0.2, inplace=True)]
			a_block = nn.Sequential(*layers)
			self.blocks.append(a_block)
		
	def forward(self, x):
		px = x
		for i in range(self.n_blocks):
			x = self.blocks[i](x)
			x = x + px
			px = x

		return x	

class cLinear(nn.Module):
	def __init__(self, in_features, out_features, bias = False):
		super(cLinear, self).__init__()

		self.in_features = in_features
		self.out_features = out_features

		j = 0
		if j == 0:
			l = -1/np.sqrt(in_features)
			h = 1/np.sqrt(in_features)
			#self.weight = nn.Parameter(1-torch.tensor(np.random.uniform(l, h, ((out_features, in_features)))).float())
			self.weight = nn.Parameter(torch.tensor(np.random.choice([0.1, 1], ((out_features, in_features)))).float())
		
		else: 
			self.weight = nn.Parameter(torch.ones((out_features, in_features)).float() * 0.1) 

		self.bias = nn.Parameter(torch.zeros(out_features).float())

	def forward(self, x):
		return F.linear(x, self.weight, self.bias)

class cLinearh(nn.Module):
	def __init__(self, in_features, out_features, bias = False):
		super(cLinearh, self).__init__()

		self.in_features = in_features
		self.out_features = out_features

		arr = np.full((out_features, in_features), 0)
		np.fill_diagonal(arr, 1)

		self.weight = nn.Parameter(torch.tensor(arr).float())
		
		

		self.bias = nn.Parameter(torch.zeros(out_features).float())

	def forward(self, x):
		return F.linear(x, self.weight, self.bias)

class Hidden_Blocks(nn.Module):
	def __init__(self, n_blocks, n_nodes, nl = True, ll = True, actvf = True):
		super(Hidden_Blocks, self).__init__()
		
		self.n_blocks = n_blocks
		self.n_nodes = n_nodes
		self.module = nn.ModuleList()

		blocks = []
		for i in range(n_blocks):
			a_block = []
			
			if nl:
				a_block.append(nn.LayerNorm(n_nodes))
			if ll:
				a_block.append(nn.Linear(n_nodes,n_nodes))
			if actvf:
				a_block.append(cLeakyReLU(n_nodes, 0.2, inplace=True)) #
				#a_block.append(nn.Sigmoid())
			
			blocks.extend(a_block)
		
		self.module.append(nn.Sequential(*blocks))

	def forward(self, x):
		x = self.module[0](x)
		return x	


class Hidden_BlocksS(nn.Module):
	def __init__(self, n_blocks, in_features, hidden_features, out_features):
		super(Hidden_BlocksS, self).__init__()
		
		self.n_blocks = n_blocks
		self.hidden_features = hidden_features

		blocks = []
		for i in range(n_blocks):
			if i == n_blocks - 1:
				a_block = nn.Sequential(
					nn.LayerNorm(hidden_features),
					nn.Linear(hidden_features, out_features, bias = False),
					nn.Tanh()
					#nn.Sigmoid()
				)

			elif i == 0:
				a_block = nn.Sequential(
					nn.Linear(in_features, hidden_features, bias = False),
					nn.LeakyReLU(0.2, inplace=True)
				)

			else:
				a_block = nn.Sequential(
					nn.LayerNorm(hidden_features),
					nn.Linear(hidden_features, hidden_features, bias = False),
					nn.LeakyReLU(0.2, inplace=True)
				)
			
			blocks.append(a_block)
		
		self.module = nn.ModuleList(blocks)

	def locked(self, x):
		noise = torch.tensor(np.random.uniform(-1, 1, (x.shape[0], self.hidden_features))).float()
		a = 1
		for i in range(self.n_blocks):
			if i != self.n_blocks - 1:
				x = a*self.module[i](x) + (1-a)*noise
			else:
				out = self.module[i](x)

		return out
	
	def train(self, x):
		for i in range(self.n_blocks):
			x = self.module[i](x)
		return x
	
	def forward(self, x):
		for i in range(self.n_blocks):
			x = self.module[i](x)
		return x

def eucli_dis(arr1, arr2):
	dis = torch.sum(torch.abs(arr1 - arr2), dim = 1)
	return dis

def get_cn(cc, epoch):
	chosen_cc = [0 for i in range(cc.shape[0])]
	scores = [0 for i in range(cc.shape[0])]

	for i in range(epoch):
		czi = torch.tensor(np.random.normal(0,1,(1,cc.shape[1]))).float()

		sv = nn.CosineSimilarity()(cc, czi)
		sv2 = torch.sort(sv, descending=True)
		index = sv2.indices[0].item()
		score = sv2.values[0].item()

		if score > scores[index]:
			scores[index] = score
			chosen_cc[index] = czi[0].numpy()

	
	chosen_cc = np.array(chosen_cc)
	chosen_cc = torch.tensor(chosen_cc).float()
	return chosen_cc, scores

def get_dist(arr):
	init = torch.zeros((1, 64))
	for i in range(arr.shape[0]):
		indices = np.random.choice(arr.shape[1], np.random.randint(0, arr.shape[1]), replace = False)
		init[0][indices] = arr[i][indices]

	return init


class socket_layer(nn.Module):
	def __init__(self, in_features, out_features, node_height = 64, node_width = 2):
		super(socket_layer, self).__init__()
		
		self.in_features = in_features
		self.out_features = out_features

		nodes = []
		for i in range(self.out_features):
			layers = []
			for j in range(node_width):

				if j == 0:
					a_block = [
						nn.LayerNorm(in_features),
						nn.Linear(in_features, node_height, bias = False),
						nn.LeakyReLU(0.2, inplace=True)
					]

				elif j == node_width - 1:
					a_block = [
						nn.LayerNorm(node_height),
						nn.Linear(node_height, 1, bias = False),
						nn.LeakyReLU(0.2, inplace=True)
					]

				else:
					a_block = [
						nn.LayerNorm(node_height),
						nn.Linear(node_height, node_height, bias = False),
						nn.LeakyReLU(0.2, inplace=True)
					]

				layers.extend(a_block)
			
			nodes.append(nn.Sequential(*layers))
				
		self.module = nn.ModuleList(nodes)

	def forward(self, x):
		outs = []
		for i in range(self.out_features):
			out = self.module[i](x)
			#print(out.shape)#; quit()
			outs.append(out)
		
		outs = torch.cat(outs, dim = 1)
		#print(outs.shape); quit()
		return outs


class cLeakyReLU(nn.Module):
	def __init__(self, in_features, negative_slope = 0.01, inplace = True):
		super(cLeakyReLU, self).__init__()
		
		self.negative_slope = negative_slope
		self.in_features = in_features
		arr = np.full((1, in_features), self.negative_slope)

		self.weight = nn.Parameter(torch.tensor(arr).float())
		

	def forward(self, x):
		jn = x.detach().numpy()
		for i in range(len(x)):
			indices = np.where(jn[i] < 0)[0]
			x[i][x[i]<0] = x[i][x[i]<0] * self.weight[0][indices]
		#print(self.weight[0][:4])
		return x	


def split_permute_images(input_images, tile_size):
    if len(input_images.shape) != 4:
        raise ValueError("Input images should be 4D (b, c, w, h).")

    b, c, w, h = input_images.shape
    w1, h1 = tile_size

    if w % w1 != 0 or h % h1 != 0:
        raise ValueError("Tile size must evenly divide the image dimensions.")

    # Calculate the number of rows and columns for the tiles
    num_rows = w // w1
    num_cols = h // h1

    # Reshape the input images to split them into tiles
    reshaped_images = input_images.reshape(b, c, num_rows, w1, num_cols, h1)

    # Flatten the tiles, permute, and reshape them
    reshaped_images = reshaped_images.transpose(0, 2, 4, 1, 3, 5)
    reshaped_images = reshaped_images.reshape(b, num_rows * num_cols, c, w1, h1)
    
    # Shuffle the tiles along axis 1 (permutation)
    for i in range(b):
        random.shuffle(reshaped_images[i])

    # Reshape the shuffled tiles to reconstruct them
    reconstructed_images = reshaped_images.reshape(b, num_rows, num_cols, c, w1, h1)
    reconstructed_images = reconstructed_images.transpose(0, 3, 1, 4, 2, 5)
    reconstructed_images = reconstructed_images.reshape(b, c, w, h)

    return reconstructed_images

def switch_tiles(batch_images, tile_size):
    if len(batch_images.shape) != 4:
        raise ValueError("Input batch should be 4D (b, c, w, h).")

    b, c, w, h = batch_images.shape
    w1, h1 = tile_size

    if w % w1 != 0 or h % h1 != 0:
        raise ValueError("Tile size must evenly divide the image dimensions.")

    # Calculate the number of rows and columns for the tiles
    num_rows = w // w1
    num_cols = h // h1

    # Reshape the images to (b, c, num_rows, w1, num_cols, h1)
    reshaped_images = batch_images.reshape(b, c, num_rows, w1, num_cols, h1)

    # Create an array to store the switched tiles
    switched_tiles = np.empty_like(reshaped_images)

    for i in range(b):
        # Create a list of all images except the current one
        other_images = [j for j in range(b) if j != i]

        for j in range(num_rows):
            for k in range(num_cols):
                # Randomly select one of the other images
                swap_image_idx = np.random.choice(other_images)

                # Swap the tiles between the current image and the selected image
                switched_tiles[i, :, j, :, k, :] = reshaped_images[swap_image_idx, :, j, :, k, :]

    # Reshape the switched tiles back to the original shape
    switched_images = switched_tiles.reshape(b, c, w, h)

    return switched_images



def find_closest_pixels(image1, images2):
    # Initialize the resulting array
    closest_pixels = np.zeros_like(image1)
    
    for pixel_idx in range(image1.shape[0]):
        pixel1 = image1[pixel_idx]
        distances = np.linalg.norm(images2[:, pixel_idx, None] - pixel1, axis=0)
        closest_pixel_idx = np.argmin(distances)
        closest_pixels[pixel_idx] = images2[closest_pixel_idx, pixel_idx]
    
    return closest_pixels

def get_data_kind(images1, images2):
	data_kinds = []
	t_images2 = images2.T
	for i in range(images1.shape[0]):
		image1 = images1[i]
		pixel_diffs = torch.abs(images2 - image1)
		t_pixel_diffs = pixel_diffs.T
		indices = torch.argmin(t_pixel_diffs, dim=1)
		data_kind = t_images2[torch.arange(images2.shape[1]),indices]
		data_kinds.append(data_kind.unsqueeze(0))
	
	data_kinds = torch.cat(data_kinds)
	return data_kinds



def get_in_data(train_imgs, bs):
    # Generate random indices for all batches at once
    indcs = np.random.randint(0, train_imgs.shape[0], size=(bs, train_imgs.shape[0]))

    # Use advanced indexing to directly extract diagonal elements
    diagonal_elements = train_imgs[indcs, np.arange(train_imgs.shape[0])]
    
    return diagonal_elements

def compare_imagesl(images1, images2):
    # Expand dimensions to make images1 (64, 1, 784)
    images1 = images1.unsqueeze(1)

    # Calculate the pixel-wise mean squared error (MSE) loss between images1 and all images in images2
    mse_losses = torch.mean((images2.unsqueeze(0) - images1) ** 2, dim=2)

    # Find the index of the image in images2 with the smallest MSE loss for each image in images1
    most_similar_indices = torch.argmin(mse_losses, dim=1)

    # Use advanced indexing to gather the most similar images from images2
    similar_images = images2[most_similar_indices]

    return similar_images

def compare_images(images1, images2):
	# Expand dimensions to make images1 (64, 1, 784)
	images1 = images1.unsqueeze(1)

	# Calculate the pixel-wise absolute difference between images1 and all images in images2
	abs_diff = torch.abs(images2.unsqueeze(0) - images1)

	# Sum the absolute differences along each pixel dimension to get the total difference
	total_diff = torch.sum(abs_diff, dim=2)
	

	# Find the index of the image in images2 with the smallest total difference for each image in images1
	most_similar_indices = torch.argmin(total_diff, dim=1)

	# Use advanced indexing to gather the most similar images from images2
	similar_images = images2[most_similar_indices]

	return similar_images

from PIL import Image
def jpg_to_grayscale_resize_and_array(image_path, target_size):
	# Open the JPG image
	img = Image.open(image_path)

	# Convert the image to grayscale
	img_gray = img.convert('L')

	# Resize the grayscale image to the specified target size with antialiasing
	img_resized = img_gray.resize(target_size, 3)  # 3 corresponds to Image.ANTIALIAS

	# Convert the resized grayscale image to a NumPy array
	img_array = np.array(img_resized)

	return img_array
    
    
   
    
def dup_elements(a):
    num_rows, num_cols = a.shape
    
    # Create an array where each element is repeated twice
    repeated_a = np.repeat(a, 2, axis=1)
    
    # Reshape the array to match the desired shape
    c = repeated_a.reshape(num_rows, num_cols * 2)

    return c








