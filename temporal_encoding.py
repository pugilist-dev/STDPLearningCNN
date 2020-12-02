# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 00:11:06 2020

@author: Chris
"""
import numpy as np
from os.path import dirname, realpath
from skimage import io
import os
from skimage.filters import difference_of_gaussians

def images_import(img_path): # returns list of images
    imgs = []    
    for img in os.listdir(img_path):    
        imgs.append(io.imread(os.path.join(img_path, img)))    
    return imgs

def temporal_encoding(image, time_window, num_layers): 
    ind = np.argsort(1 / image.flatten())
    lat = np.sort(1 / image.flatten())
    ind = np.delete(ind, np.where(lat == np.inf))
    
    ind_mod = np.unravel_index(ind, image.shape)
    t_step = np.ceil(np.arange(ind.size) / ((ind.size) / (time_window - num_layers))).astype(np.uint8)
    ind_mod += (t_step,)
    spikes = np.zeros((image.shape[0], image.shape[1], time_window))
    spikes[ind_mod] = 1
    return spikes

# Demonstration

# define path of the test set
test_path = os.path.dirname(dirname(realpath(__file__))) + '\BIC Project\images'

# import test set
test_set=images_import(test_path)

# display example image
import matplotlib.pyplot as plt
plt.imshow(test_set[0], cmap='gray')
plt.show()

# apply DoG filter
dog = difference_of_gaussians(test_set[0], 1.5)

# display example image after filter
plt.imshow(dog, cmap='gray')
plt.show()

# temporal encoding example
window = 5
layers = 1
temp_test=temporal_encoding(dog, window, layers)

for i in range(window):
    plt.imshow(temp_test[:,:,i], cmap='Greys')
    plt.show()