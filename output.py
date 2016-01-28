# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 23:27:01 2016

@author: channerduan
"""
from numpy import genfromtxt, savetxt
import numpy as np

dataset = genfromtxt(open('submission2.csv','r'), delimiter=',', dtype='f8')[0:]
size = np.size(dataset) + 1;
print dataset.size
index = np.arange(1, size);
combine = (np.concatenate((index, dataset), axis=0));
result = combine.reshape(2, size-1);
print dataset.shape
print index.shape
print combine.shape
print result.shape

aa = np.c_[index, dataset]
print help(np.c_)

savetxt('submission.csv', result.T, delimiter=',', fmt='%d', header='ImageId,Label', comments='')

