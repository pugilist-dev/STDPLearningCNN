#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:17:44 2020
@author: rajiv
"""

import numpy as np
import os
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from DoG import DoG
from tqdm import trange
from numba import *
from numba import cuda
from math import floor, ceil
from conv import conv1, conv2, conv3, pool1, pool2
from utils import freq, get_dim, init_layers, lateral_inh

train_data = '/home/rajiv/Documents/lectures/BIC/project_conv_sdnn/datasets/TrainingSet/Face/'
test_data = '/home/rajiv/Documents/lectures/BIC/project_conv_sdnn/datasets/TestingSet/Face'

network_params = [
                    {'Type': 'conv', 'num_filters': 4, 'filter_size': 5, 'thresh': 10., 'stride': 1},
                    {'Type': 'pool', 'num_filters': 4, 'filter_size': 7, 'thresh': 0., 'stride': 6},
                    {'Type': 'conv', 'num_filters': 20, 'filter_size': 17, 'thresh': 60., 'stride':1},
                    {'Type': 'pool', 'num_filters': 20, 'filter_size': 5, 'thresh': 0., 'stride': 5},
                    {'Type': 'conv', 'num_filters': 20, 'filter_size': 5, 'thresh': 2., 'stride':1}
                ]

max_learn_iter = [3000, 0, 5000, 0, 6000, 0]

stdp_params = {
               'max_learn_iter': max_learn_iter,
               'stdp_per_layer': [10, 0, 4, 0, 2],
               'max_iter': sum(max_learn_iter),
               'a_minus': np.array([.003, 0, .0003, 0, .0003], dtype=np.float32),
               'a_plus': np.array([.004, 0, .0004, 0, .0004], dtype=np.float32)
               }
image_size = (400, 400)
win = [0,15]

   
layers = []
total_time = 15
layer_1 = conv1().to('cuda')
layers.append(init_layers(total_time = total_time, C = 4, H = 400, W = 400))
layer_2 = pool1().to('cuda')
layers.append(init_layers(total_time = total_time, C = 4, H = 66, W = 66))
layer_3 = conv2().to('cuda')
layers.append(init_layers(total_time = total_time, C = 20, H = 54, W = 54))
layer_4 = pool2().to('cuda')
layers.append(init_layers(total_time = total_time, C = 20, H = 10, W = 10))
layer_5 = conv3().to('cuda')
layers.append(init_layers(total_time = total_time, C = 20, H = 6, W = 6))
network = [layer_1, layer_2, layer_3, layer_4, layer_5]
network_len = len(network)
layer_filt_dimension = get_dim(network[0::2])

thresh = torch.Tensor([10., 60., 2. ])
image_names = os.listdir(train_data)
data_len = len(image_names)
thds_per_dim = 10

for j in trange(data_len):
    image = DoG(image_name = image_names[j])
    image = np.reshape(image, (160000,))
    # code to change the image to total_time 
    image = freq(15, image, 0.05)
    # Codde ends here
    image = torch.from_numpy(image).to('cuda')
    image = image.float()
    
    #for i in range(stdp_params['max_iter']):
    for t in range(total_time):
        ## Slice the image of the first time step
        for l in range(network_len):
            V = layers[l]['V'][t-1, :, :, :]
            S = layers[l]['S'][t, :, :, :]
            K_inh =layers[l]['K_inh']
            if network[l].__class__.__name__ == "conv1":
                img = image[t,:,:,:].unsqueeze(0)
                V = network[l](img)
                V = V.permute(0, 3, 2, 1)
                S = (V > torch.Tensor([10.0]).to('cuda')).float()*1
                N, H, W, C = V.shape
                layers[l]['V'][t,:,:,:] = V[0]

                S = S.cpu().data.numpy()
                V = V.cpu().data.numpy()
                K_inh = K_inh.cpu().data.numpy()
                S, K_inh = lateral_inh(S[0], V[0], K_inh)
                S = torch.from_numpy(S)
                S = S.float()
                K_inh = torch.from_numpy(K_inh)
                K_inh = K_inh.float()
                layers[l]['S'][t,:,:,:] = S
                layers[l]['K_inh'] = K_inh
            elif network[l].__class__.__name__ == "pool1":
                S = layers[l-1]['S'][t,:,:,:].unsqueeze(0)
                S = S.permute(0,3,1,2).to('cuda')
                S = network[l](S)
                S = S.float()
            elif network[l].__class__.__name__ == "conv2":
                S = layers[l-1]['S'][t,:,:,:].unsqueeze(0)
                S = S.permute(0,3,1,2).to('cuda')
                V = network[l](S)
                V = V.permute(0, 3, 2, 1)
                S = (V > torch.Tensor([10.0]).to('cuda')).float()*1
                N, C, H, W = V.shape
                layers[l]['V'][t,:,:,:] = V[0]
                S = S.cpu().data.numpy()
                V = V.cpu().data.numpy()
                K_inh = K_inh.cpu().data.numpy()
                S, K_inh = lateral_inh(S[0], V[0], K_inh)
                S = torch.from_numpy(S)
                S = S.float()
                K_inh = torch.from_numpy(K_inh)
                K_inh = K_inh.float()
                layers[l]['S'][t,:,:,:] = S
                layers[l]['K_inh'] = K_inh
            elif network[l].__class__.__name__ == "pool2":
                S = layers[l-1]['S'][t,:,:,:].unsqueeze(0)
                S = S.permute(0,3,1,2).to('cuda')
                S = network[l](S)
                S = S.float()
            elif network[l].__class__.__name__ == "conv3":
                S = layers[l-1]['S'][t,:,:,:].unsqueeze(0)
                S = S.permute(0,3,1,2).to('cuda')
                V = network[l](S)
                V = V.permute(0, 3, 2, 1)
                S = (V > torch.Tensor([10.0]).to('cuda')).float()*1
                N, C, H, W = V.shape
                layers[l]['V'][t,:,:,:] = V[0]
                S = S.cpu().data.numpy()
                V = V.cpu().data.numpy()
                K_inh = K_inh.cpu().data.numpy()
                S, K_inh = lateral_inh(S[0], V[0], K_inh)
                S = torch.from_numpy(S)
                S = S.float()
                K_inh = torch.from_numpy(K_inh)
                K_inh = K_inh.float()
                layers[l]['S'][t,:,:,:] = S
                layers[l]['K_inh'] = K_inh
            else:
                print("The layer does not exist")
        ### start the STDP Learning 