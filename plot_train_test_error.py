# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 22:08:28 2021

@author: kbdra
"""
from matplotlib import pyplot as plt
import numpy as np


p_train = np.array([1/2, 1/4, 1/8, 1/16])
p_train = np.log(np.multiply(p_train, 51004))

train_loss = np.log( [0.0609, 0.0906, 0.1144, 0.1756])
train_accuracy = [25026/25502, 12396/12751, 6153/6375, 3015/3187]
train_error = 1 - np.array(train_accuracy)
train_error = np.log(train_error)

test_loss = np.log([0.0656, 0.1045, 0.1354, 0.2069])
test_accuracy = [9802/10000, 9684/10000, 9600/10000, 9401/10000]
test_error = 1 - np.array(test_accuracy)
test_error = np.log(test_error)


plt.plot(p_train, train_error)
plt.plot(p_train, test_error)
plt.xlabel('log(training_set points)')
plt.ylabel('log(error)')
plt.legend(('train', 'test'))