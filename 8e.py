# -*- coding: utf-8 -*-
"""
Created on Sat May  1 02:39:07 2021

@author: kbdra
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

import os


### part 8e, i

test_dataset = datasets.MNIST('./data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

device = torch.device("cpu")
model = Net().to(device)


feature = []
label = []
#get features
for test_data in test_dataset:
    arr = model.features(torch.reshape(test_data[0], (1,1,28,28))).detach().numpy()
    feature.append(arr[0])
    label.append(test_data[1])

feature = np.array(feature)

tsne = TSNE(2)
result = tsne.fit_transform(feature)

tsne1 = result.T[1]
tsne0 = result.T[0]
sns.scatterplot(x=tsne0,y=tsne1,hue=label, s= 2)


## part 8e, ii

results = np.zeros((4*28, 8*28))

for i in range(4):
    feature0 = feature[np.random.randint(0, len(feature))]
    inds = []
    for k in range(8):
        ind = 0
        smallest = 1000
        smallest_ind = []
        for sample_feature in feature:
            if (np.linalg.norm(feature0 - sample_feature) <= smallest) and not(ind in inds):
                smallest = np.linalg.norm(feature0 - sample_feature)
                smallest_ind = ind
            ind += 1
        inds.append(smallest_ind)
        results[i*28:(i+1)*28, k*28:(k+1)*28] = test_dataset[smallest_ind][0]

plt.imshow(results)
plt.show()






