#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:03:03 2020

@author: rajiv
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

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
        self.conv = nn.Conv2d(in_channels = 20, out_channels = 20, kernel_size =(5, 5), stride= 1, padding = 0, bias = False)
        
    def forward(self, x):
        out = self.conv(x)
        return out
    