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
from utils import freq, get_dim, init_layers, lateral_inh, get_STDP_idxs, STDP_learning, get_weights, save_weights

train_data = '/home/rajiv/Documents/lectures/BIC/project_conv_sdnn/datasets/TrainingSet/Face/'
test_data = '/home/rajiv/Documents/lectures/BIC/project_conv_sdnn/datasets/TestingSet/Face'
max_learn_iter = [10, 10, 10]

network_params = [    {'Type': 'conv', 'num_filters': 4, 'filter_size': 5, 'th': 10.},
                      {'Type': 'pool', 'num_filters': 4, 'filter_size': 7, 'th': 0., 'stride': 6},
                      {'Type': 'conv', 'num_filters': 20, 'filter_size': 17, 'th': 60.},
                      {'Type': 'pool', 'num_filters': 20, 'filter_size': 5, 'th': 0., 'stride': 5},
                      {'Type': 'conv', 'num_filters': 20, 'filter_size': 5, 'th': 2.}]

stdp_params = {
               'max_learn_iter': max_learn_iter,
               'stdp_per_layer': [10, 4, 2],
               'max_iter': sum(max_learn_iter),
               'a_minus': np.array([3.,6.,9.], dtype=np.float32),
               'a_plus': np.array([4.,6.,8.], dtype=np.float32),
               'offset_STDP': [floor(network_params[0]['filter_size']),
                                   floor(network_params[2]['filter_size']/8),
                                   floor(network_params[4]['filter_size'])]
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

thresh = torch.Tensor([.5, .5, .5 ])
image_names = os.listdir(train_data)
data_len = len(image_names)
thds_per_dim = 10

for j in trange(data_len):
    image = DoG(image_name = image_names[j])
    image = np.reshape(image, (160000,))
    image = freq(15, image, 0.05)
    image = torch.from_numpy(image).to('cuda')
    image = image.float()
    
    for i in range(stdp_params['max_iter']):
        for t in range(total_time):
            for l in range(network_len):
                V = layers[l]['V'][t-1, :, :, :]
                S = layers[l]['S'][t, :, :, :]
                K_inh =layers[l]['K_inh']
                if network[l].__class__.__name__ == "conv1":
                    img = image[t,:,:,:].unsqueeze(0)
                    V = network[l](img)
                    V = V.permute(0, 3, 2, 1)
                    S = (V > torch.Tensor([.5]).to('cuda')).float()*1
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
                    K_STDP = layers[l]['K_STDP']
                    valid = S*V[0]*K_STDP
                    valid = valid.cpu().data.numpy()
                    maxval, maxind1, maxind2 = get_STDP_idxs(valid, H = H, W = W, D = C, 
                                                             layer_idx = l, offset = stdp_params['offset_STDP'][0],
                                                             stdp_per_layer = stdp_params['stdp_per_layer'][0])
                    w = get_weights(network[l])
                    Weight, K_STDP = STDP_learning(S_sz = S.shape, s = S, w = w, K_STDP = K_STDP, maxval = maxval,
                                              maxind1 = maxind1, maxind2 = maxind2, stride = 1, offset = stdp_params['offset_STDP'][0],
                                              a_minus = stdp_params['a_minus'][0], 
                                              a_plus = stdp_params['a_plus'][0])
                    save_weights(network[l], Weight)
                    layers[l]['K_STDP'] = K_STDP
                        
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
                    S = (V > torch.Tensor([.5]).to('cuda')).float()*1
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
                    K_STDP = layers[l]['K_STDP']
                    valid = S*V[0]*K_STDP
                    valid = valid.cpu().data.numpy()
                    maxval, maxind1, maxind2 = get_STDP_idxs(valid, H = H, W = W, D = C, 
                                                             layer_idx = l, offset = stdp_params['offset_STDP'][1],
                                                             stdp_per_layer = stdp_params['stdp_per_layer'][1])
                    w = get_weights(network[l])
                    Weight, K_STDP = STDP_learning(S_sz = S.shape, s = S, w = w, K_STDP = K_STDP, maxval = maxval,
                                              maxind1 = maxind1, maxind2 = maxind2, stride = 1, offset = stdp_params['offset_STDP'][1],
                                              a_minus = stdp_params['a_minus'][1], a_plus = stdp_params['a_plus'][1])
                    save_weights(network[l], Weight)
                    layers[l]['K_STDP'] = K_STDP
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
                    S = (V > torch.Tensor([.5]).to('cuda')).float()*1
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
                    K_STDP = layers[l]['K_STDP']
                    valid = S*V[0]*K_STDP
                    valid = valid.cpu().data.numpy()
                    maxval, maxind1, maxind2 = get_STDP_idxs(valid, H = H, W = W, D = C, 
                                                             layer_idx = l, offset = stdp_params['offset_STDP'][2],
                                                             stdp_per_layer = stdp_params['stdp_per_layer'][2])
                    w = get_weights(network[l])
                    layers[l]['K_STDP'] = K_STDP
                else:
                    print("The layer does not exist")
path1 = "./layer_1.pth"
path2 = "./layer_2.pth"
path3 = "./layer_3.pth"
path4 = "./layer_4.pth"

torch.save(layer_1.state_dict(), path1)
torch.save(layer_2.state_dict(), path2)
torch.save(layer_2.state_dict(), path3)
torch.save(layer_2.state_dict(), path4)

class perceptron(torch.nn.Module):
    def __init__(self):
        super(perceptron, self).__init__()
        self.fc1 = nn.Linear(2000, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 29)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        out = F.softmax(x)
        return out
    
model = perceptron()
model.to('cuda')
criterion = nn.CrossEntropyLoss().to('cuda')
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
path = "/home/rajiv/Documents/lectures/BIC/project_conv_sdnn/datasets/TrainingSet/faces_labeled/"
classes = os.listdir(path)

test_images = {"images": [],
               "label": []
               }
for i in classes:
    test_images["images"].append(os.listdir(path+i))
    test_images["label"].append(i)


j = 0
losses = []
accuracy = []
for c in test_images['label']:
    label = int(c)
    label = [label]*15
    label = torch.Tensor(label)
    label = label.to('cuda')
    label = label.long()
    for i in test_images['images'][j]:
        image_name = c + '/' + i
        image = DoG(image_name = image_name)
        image = np.reshape(image, (160000,))
        image = freq(15, image, 0.05)
        image = torch.from_numpy(image).to('cuda')
        image = image.float()
        image = layer_1(image)
        image = layer_2(image)
        image = layer_3(image)
        image = layer_4(image)
        image = torch.reshape(image, (15, 2000))
        optimizer.zero_grad()
        prediction = model(image)
        _,preds = torch.max(prediction, 1)
        acc = (torch.argmax(preds) == int(c))*1
        accuracy.append(acc.cpu().data.numpy())
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()
        losses.append(loss)
        
    j+=1
path = "/home/rajiv/Documents/lectures/BIC/project_conv_sdnn/datasets/TestingSet/faces_labeled/"
classes = os.listdir(path)

test_images = {"images": [],
               "label": []
               }
for i in classes:
    test_images["images"].append(os.listdir(path+i))
    test_images["label"].append(i)
    
j = 0
losses = []
accuracy = []
for c in test_images['label']:
    label = int(c)
    label = [label]*15
    label = torch.Tensor(label)
    label = label.to('cuda')
    label = label.long()
    for i in test_images['images'][j]:
        image_name = c + '/' + i
        image = DoG(image_name = image_name, Test = True)
        image = np.reshape(image, (160000,))
        image = freq(15, image, 0.05)
        image = torch.from_numpy(image).to('cuda')
        image = image.float()
        image = layer_1(image)
        image = (image > torch.Tensor([.5]).to('cuda')).float()*1
        image = layer_2(image)
        image = layer_3(image)
        image = (image > torch.Tensor([.5]).to('cuda')).float()*1
        image = layer_4(image)
        image = torch.reshape(image, (15, 2000))
        prediction = model(image)
        _,preds = torch.max(prediction, 1)
        acc = (torch.argmax(preds) == int(c))*1
        accuracy.append(acc.cpu().data.numpy())
        loss = criterion(prediction, label)
        losses.append(loss)
        
    j+=1
    
print("The accuracy is: ",sum(accuracy)/len(accuracy))