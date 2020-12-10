#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:09:28 2020

@author: rajiv
"""
import numpy as np
import torch
from torch import nn
import torchvision
from numba import jit
import itertools

win = [0,15]
def freq(tw,x,fmax): 
    #provide a time window, flattened image (U_train[0]), and max frequency
    out=np.zeros((len(x),int(win[1]-win[0]))) # initialize output matrix
    spikes = []
    for i in range(len((x))): # for each pixel in image
        spikes = round(((float((win[1]-win[0])))*fmax)*x[i]) # calculate the number of spikes
        spikeprob=spikes/float((win[1]-win[0])) # calculate the spike probability
        out[i]=np.random.choice(np.arange(0, 2), size=(win[1]-win[0]), replace=True, p=[1-spikeprob, spikeprob])
        # randomly populate output matrix with spikes according to spike probability
    out = np.reshape(out, (15, 1, 400, 400))
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
    d_tmp['S'] = torch.from_numpy(np.zeros((total_time, H, W, C)).astype(np.float32))
    d_tmp['V'] = torch.from_numpy(np.zeros((total_time, H, W, C)).astype(np.float32))
    d_tmp['K_STDP'] = torch.from_numpy(np.ones((H, W, C)).astype(np.float32))
    d_tmp['K_inh'] = torch.from_numpy(np.ones((H, W)).astype(np.float32))
    return d_tmp

@jit
def lateral_inh(S, V, K_inh):
    S_inh = np.ones(S.shape, dtype=S.dtype)
    K = np.ones(K_inh.shape, dtype=K_inh.dtype)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            for k in range(V.shape[2]):
                flag = False
                if S[i, j, k] != 1:
                    continue
                if K_inh[i, j] == 0:
                    S_inh[i, j, k] = 0
                    continue
                for kz in range(V.shape[2]):
                    if S[i, j, kz] == 1 and V[i, j, k] < V[i, j, kz]:
                        S_inh[i, j, k] = 0
                        flag = True
                if flag:
                    continue
                else:
                    K[i, j] = 0
    S *= S_inh
    K_inh *= K
    return S, K_inh

@jit
def get_STDP_idxs(valid, H, W, D,offset, layer_idx, stdp_per_layer):
        i = layer_idx
        STDP_counter = 1
        mxv = np.amax(valid, axis=2)
        mxi = np.argmax(valid, axis=2)
        maxind1 = np.ones((D, 1)) * -1
        maxind2 = np.ones((D, 1)) * -1
        maxval = np.ones((D, 1)) * -1
        while np.sum(np.sum(mxv)) != 0.:
            if STDP_counter > stdp_per_layer:
                break
            else:
                STDP_counter += 1
            maximum = np.amax(mxv, axis=1)
            index = np.argmax(mxv, axis=1)
            index1 = np.argmax(maximum)
            index2 = index[index1]
            maxval[mxi[index1, index2]] = mxv[index1, index2]
            maxind1[mxi[index1, index2]] = index1
            maxind2[mxi[index1, index2]] = index2
            mxv[mxi == mxi[index1, index2]] = 0
            mxv[max(index1 - offset, 0):min(index1 + offset, H) + 1,
                max(index2 - offset[layer_idx], 0):min(index2 + offset, W) + 1] = 0
        maxval = np.squeeze(maxval).astype(np.float32)
        maxind1 = np.squeeze(maxind1).astype(np.int16)
        maxind2 = np.squeeze(maxind2).astype(np.int16)
        return maxval, maxind1, maxind2

@jit
def STDP_learning(S_sz, s, w, K_STDP,
                  maxval, maxind1, maxind2,
                  stride, offset, a_minus, a_plus):
    for idx in range(S_sz[0]):
        for idy in range(S_sz[1]):
            for idz in range(S_sz[2]):
                if idx != maxind1[idz] or idy != maxind2[idz]:
                    continue
                for i in range(w.shape[3]):
                    if (idz != i and maxind1[idz] <= maxind1[i] + offset
                        and maxind1[idz] >= maxind1[i] - offset
                        and maxind2[idz] <= maxind2[i] + offset
                        and maxind2[idz] >= maxind2[i] - offset
                        and maxval[i] > maxval[idz]):
                        maxval[idz] = 0.
                if maxval[idz] > 0:
                    input = np.zeros(w[:, :, :, idz].shape)
                    if idy*stride >= S_sz[1] - w.shape[1] and idx*stride >= S_sz[0] - w.shape[0]:
                        ss = s[idx * stride:, idy * stride:, :]
                        input[:ss.shape[0], :ss.shape[1], :] = ss
                    elif idy*stride >= S_sz[1] - w.shape[1]:
                        ss = s[idx * stride:idx * stride + w.shape[0], idy * stride:, :]
                        input[:, :ss.shape[1], :] = ss
                    elif idx*stride >= S_sz[0] - w.shape[0]:
                        ss = s[idx * stride:, idy * stride:idy * stride + w.shape[1], :]
                        input[:ss.shape[0], :, :] = ss
                    else:
                        input = s[idx * stride:idx*stride+w.shape[0], idy*stride:idy*stride+w.shape[1], :]
                    dw = input * a_minus * w[:, :, :, idz] * (1 - w[:, :, :, idz]) + \
                         input * a_plus * w[:, :, :, idz] * (1 - w[:, :, :, idz]) - \
                         a_minus * w[:, :, :, idz] * (1 - w[:, :, :, idz])
                    w[:, :, :, idz] += dw
                    for k in range(S_sz[2]):
                        j = 0 if idy - offset < 0 else idy - offset
                        while j <= (S_sz[1] - 1 if idy + offset > S_sz[1] - 1 else idy + offset):
                            i = 0 if idx - offset < 0 else idx - offset
                            while i <= (S_sz[0] - 1 if idx + offset > S_sz[0] - 1 else idx + offset):
                                K_STDP[i, j, k] = 0
                                i += 1
                            j += 1
                    for j in range(S_sz[1]):
                        for i in range(S_sz[0]):
                            K_STDP[i, j, idz] = 0
    return w, K_STDP

def get_weights(weights):
    
    iterator = itertools.islice(weights.state_dict().items(), 0,1)
    key_values = next(iterator)
    weight = key_values[1].permute(2,3,0,1)
    w = weight.cpu().data.numpy()
    return w

def save_weights(layer, weights):
    weights = torch.from_numpy(weights).to('cuda')
    weights = weights.permute(2,3,0,1)
    iterator = itertools.islice(layer.state_dict().items(), 0,1)
    key_values = next(iterator)
    for i in range(len(key_values[1])):
        key_values[1][i] = weights[i]
    return