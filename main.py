#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:17:44 2020

@author: rajiv
"""

import numpy as np
from os.path import dirname , realpath
from math import floor
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

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
image_size = (250, 160)
class conv1(nn.Module):
    def __init__(self):
        super(conv1, self).__init__()
        self.conv =  nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = (5,5), stride=1, padding = 0, bias=False)
    
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
        self.conv = nn.Conv2d(in_channels = 4, out_channels = 20, kernel_size = (17, 17), stride = 1, padding = 0, bias = False)
        
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
        self.conv = nn.Conv2d(in_channels = 20, out_channels = 20, kernel_size =(5, 5), stride= 1, padding = 0, bias = False)
        
    def forward(self, x):
        out = self.conv(x)
        return out
    
layer_1 = conv1()
layer_2 = pool1()
layer_3 = conv2()
layer_4 = pool2()
layer_5 = conv3()