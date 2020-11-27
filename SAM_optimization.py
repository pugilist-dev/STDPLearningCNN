#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 21:34:26 2020

@author: rajiv
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 02:38:36 2020

@author: rajiv
"""

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torchvision
from torch.utils.data import DataLoader
import json
from PIL import Image
from imgaug import augmenters as iaa
from initialize import initialize
from step_lr import StepLR
from log import Log
from smooth_cross_entropy import smooth_crossentropy
import argparse
import my_model as mm
from wide_res_net import WideResNet
image_size = (224, 224)
batch_size = 20

class ImgAugTransform:
  def __init__(self):
    self.aug = iaa.Sequential([
        #iaa.Resize(image_size),
        iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-20, 20), mode='symmetric'),
        iaa.Sometimes(0.25,
                      iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                 iaa.CoarseDropout(0.1, size_percent=0.5)])),
        iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
    ])
      
  def __call__(self, img):
    img = np.array(img)
    return self.aug.augment_image(img)

transform = torchvision.transforms.Compose([
    ImgAugTransform(),
    lambda x: torch.from_numpy(x),
    torchvision.transforms.RandomVerticalFlip()
])

class dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, labels):
        self.labels = labels
        self.filenames = image_path
        self.transform = transform
        
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = Image.open(self.filenames[index])
        image = self.transform(image)
        image = image.float()
        return image.transpose(2,0), self.labels[index]
    
    
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't work like the other optimizers, you should first call `first_step` and the `second_step`; see the documentation for more info.")

    def _grad_norm(self):
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
image_path = '/home/rajiv/Documents/experiments/cassava/cassava-leaf-disease-classification/train_images/'
train = pd.read_csv('/home/rajiv/Documents/experiments/cassava/cassava-leaf-disease-classification/train.csv')
test_images = os.listdir('/home/rajiv/Documents/experiments/cassava/cassava-leaf-disease-classification/test_images/')
label_to_disease = json.load(open('/home/rajiv/Documents/experiments/cassava/cassava-leaf-disease-classification/label_num_to_disease_map.json'))
train['disease'] = train.label.map(label_to_disease)
train.label = train.label.astype(int)


train_images_path = image_path+train.image_id
train = train.drop(['image_id'], axis = 1)
train = pd.DataFrame({'image_id':train_images_path, 'label': train.label, 'disease':train.disease})
train = train.drop(['disease'], axis = 1)
train_image = train.image_id.values.tolist()
train_labels = train.label.values.tolist()

dataset = dataset(train_image,train_labels)

train_len = int(0.8*dataset.__len__())
val_len = dataset.__len__() - train_len
train, val = torch.utils.data.random_split(dataset, lengths= [train_len, val_len])
loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers = 2)
val_loader = DataLoader(val, batch_size = batch_size, shuffle=True, num_workers = 2)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=batch_size, type=int, help="Batch size used in the training and validation loop.")
parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
parser.add_argument("--epochs", default=20, type=int, help="Total number of epochs.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
parser.add_argument("--learning_rate", default=0.005, type=float, help="Base learning rate at the start of the training.")
parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
parser.add_argument("--momentum", default=0.6, type=float, help="SGD Momentum.")
parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
args = parser.parse_args()

initialize(args, seed=42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log = Log(log_each=10)

model = model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=5).to(device)
base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

for epoch in range(args.epochs):
    model.train()
    model.to(device)
    log.train(len(loader))
    for image, label in loader:
        image = image.float()
        inputs = image.to(device)
        targets = label.to(device)
        predictions = model(inputs)
        loss = smooth_crossentropy(predictions, targets)
        loss.mean().backward()
        optimizer.first_step(zero_grad=True)
        # second forward-backward step
        smooth_crossentropy(model(inputs), targets).mean().backward()
        optimizer.second_step(zero_grad=True)

        with torch.no_grad():
            correct = torch.argmax(predictions.data, 1) == targets
            log(model, loss.cpu(), correct.cpu(), scheduler.lr())
            scheduler(epoch)
        
        model.eval()
        log.eval(len_dataset=len(val_loader))
        
    with torch.no_grad():
        for img, lbl in val_loader:
            image = img.float()
            inputs = image.to(device)
            targets = lbl.to(device)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            correct = torch.argmax(predictions, 1) == targets
            log(model, loss.cpu(), correct.cpu())
    log.flush()


predict = []

for i in test_images:
    image = Image.open(f'/home/rajiv/Documents/experiments/cassava/cassava-leaf-disease-classification/test_images/{i}')
    image = image.resize(image_size)
    
    image = np.asarray(image) / 255.
    image = np.expand_dims(image, axis=0)
    
    predict.append(np.argmax(model(image)))
    

submission = pd.DataFrame({'image_id': test_images, 'label': predict})
submission.to_csv('submission.csv', index=None)