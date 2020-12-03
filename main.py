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


class conv1(nn.Module):
    def __init__(self):
        super(conv1, self).__init__()
        self.conv =  nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = (5,5), stride=1, padding = 2, bias=False)
    
    def forward(self, x):
        out = self.conv(x)
        return out

class pool1(nn.Module):
    def __init__(self):
        super(pool1, self).__init__()
    def forward(self, x):
        out = F.max_pool2d(x, kernel_size = 7, stride = 6) 
        return out

class conv2(nn.Module):
    def __init__(self):
        super(conv2, self).__init__()
        self.conv = nn.Conv2d(in_channels = 4, out_channels = 20, kernel_size = (17, 17), stride = 1, padding = 2, bias = False)
        
    def forward(self, x):
        out = self.conv(x)
        return out
    
class pool2(nn.Module):
    def __init__(self):
        super(pool2, self).__init__()
    def forward(self,x):
        out = F.max_pool2d(x, kernel_size = 5, stride = 5)
        return out
    
class conv3(nn.Module):
    def __init__(self):
        super(conv3, self).__init__()
        self.conv = nn.Conv2d(in_channels = 20, out_channels = 20, kernel_size =(5, 5), stride= 1, padding = 2, bias = False)
        
    def forward(self, x):
        out = self.conv(x)
        return out
    
    
def get_dim(layer):
    # 
    temp = []
    layer_len = len(layer)
    for i in range(layer_len):
        dim =  [i for i in layer[i].parameters()]
        N, C, H, W = dim[0].shape
        layer_index = [N, C, H, W]
        temp.append(layer_index)
    return temp

def init_layers(H, W, C, total_time):
    """
        Initialise layers         
    """
    d_tmp = {}
    H, W, C = H, W, C
    d_tmp['S'] = np.zeros((H, W, C, total_time)).astype(np.uint8)
    d_tmp['V'] = np.zeros((H, W, C, total_time)).astype(np.float32)
    d_tmp['K_STDP'] = np.ones((H, W, C)).astype(np.uint8)
    d_tmp['K_inh'] = np.ones((H, W)).astype(np.uint8)
    layers.append(d_tmp)
    return
    
layers = []
total_time = 15
layer_1 = conv1()
layers.append(init_layers(400, 400, 4, total_time))
layer_2 = pool1()
layers.append(init_layers(66, 66, 4, total_time))
layer_3 = conv2()
layers.append(init_layers(54, 54, 20, total_time))
layer_4 = pool2()
layers.append(init_layers(10, 10, 20, total_time))
layer_5 = conv3()
layers.append(init_layers(7, 8, 20, total_time = 15))
network = [layer_1, layer_2, layer_3, layer_4, layer_5]
network_len = len(network)
layer_filt_dimension = get_dim(network[0::2])

thresh = torch.Tensor([10., 60., 2. ])
image_names = os.listdir(train_data)
data_len = len(image_names)
thds_per_dim = 10

for j in trange(data_len):
    image = DoG(image_name = image_names[j])
    # code to change the image to total_time 
    
    # Codde ends here
    image = torch.from_numpy(image).to('cuda')
    image = image.float()
    
    for i in range(stdp_params['max_iter']):
        for t in range(total_time):
            ## Slice the image of the first time step
            for l in range(network_len):
                if network[l].__class__.__name__ == "conv1":
                    V = network[l](image)
                    C, H, W = 4, 400, 400 
                    S = (V > thresh[0]) * 1
                    ## code for lateral inhibition starts here##
                    S = lateral_inh()
                    ## Code for lateral inhibition end here   ##
                    S = S.float()
                elif network[l].__class__.__name__ == "pool1":
                    S = network[l](S)
                    ## code for lateral inhibition starts here##
                    
                    ## Code for lateral inhibition end here   ##
                    S = S.float()
                elif network[l].__class__.__name__ == "conv2":
                    V = network[l](S)
                    C, H, W = 20, 54, 54
                    S = (V > thresh[1]) * 1
                    ## code for lateral inhibition starts here##
                    
                    ## Code for lateral inhibition end here   ##
                    S = S.float()
                elif network[l].__class__.__name__ == "pool2":
                    S = network[l](S)
                    ## code for lateral inhibition starts here##

                    ## Code for lateral inhibition end here   ##
                    S = S.float()
                elif network[l].__class__.__name__ == "conv3":
                    V = network[l](S)
                    C, H, W = 20, 7, 8
                    S = (V > thresh[2]) * 1
                    ## code for lateral inhibition starts here##
                    
                    ## Code for lateral inhibition end here   ##
                    S = S.float()
                else:
                    print("The layer does not exist")
            ### start the STDP Learning here ###
        