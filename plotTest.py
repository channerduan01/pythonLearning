# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 19:56:18 2016

@author: channerduan
"""

from PIL import Image
from pylab import *
from scipy.ndimage import filters
import matplotlib.pyplot as plt


# read image to array
im = array(Image.open('test.png'))
# plot the image
imshow(im)
im2 = zeros(im.shape)
for i in range(3): im2[:,:,i] = filters.gaussian_filter(im[:,:,i],1)
im2 = uint8(im2)
imshow(im2)


image_file = Image.open("test.png") # open colour image
plt.gray()
image_file = image_file.convert('L') # convert image to black and white
plt.imshow(image_file)


