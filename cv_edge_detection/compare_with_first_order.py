# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:17:55 2016

@author: yx
"""

import math

from PIL import Image
import numpy as np
from scipy import signal
from scipy.ndimage import filters
import matplotlib.pyplot as plt

im = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 255, 0, 0, 0, 0, 0, 255, 0],
    [0, 0, 255, 255, 0, 255, 255, 0, 0],
    [0, 0, 255, 0, 0, 0, 255, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 255, 0, 0, 0, 255, 0, 0],
    [0, 0, 255, 255, 0, 255, 255, 0, 0],
    [0, 255, 0, 0, 0, 0, 0, 255, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
])

sigma = 0.5
temp_size = 3

    
# Laplacian mask
laplacian_mask = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

laplacian_mask2 = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
])

#Gaussian filter smooth the image
def smooth(im, sigma):
    smoothed = filters.gaussian_filter(im, np.sqrt(sigma))
    return smmothed     
      

#for the large size of the laplacian mask
def laplacian_template(temp_size):
    laplacian_mask_ = np.zeros((temp_size, temp_size))
    for n in range(temp_size):
        for m in range(temp_size):
            if n != (temp_size-1)/2 or m != (temp_size-1)/2:
                  laplacian_mask_[n][m] = 1
            elif n == (temp_size -1)/2 and m == (temp_size-1)/2:
                laplacian_mask_[n][m] = -(temp_size*temp_size - 1)
    return laplacian_mask_            

#laplacian and gaussian together
def LoG(temp_size, sigma,im):
    laplacian_temp = np.zeros((temp_size, temp_size))
    cx = (temp_size-1)/2
    cy = (temp_size-1)/2
    sum = 0
    for p in range(temp_size):
        for q in range(temp_size):
            nx = p - cx
            ny = q - cy
            laplacian_temp[q][p] = ((((((nx*nx)+(ny*ny))/(sigma*sigma))-2))*math.exp(-((nx*nx)+(ny*ny))/(2*(sigma*sigma))))/(sigma*sigma)
            sum = sum + laplacian_temp[q][p]
        
    laplacian_temp = laplacian_temp/sum
    laplacian = signal.convolve2d(im, laplacian_temp, mode = 'same')
    return laplacian
    
#zero crossing 
def zero_crossing(laplacian):
    M,N = laplacian.shape
    edge = np.zeros(laplacian.shape)
    for (y, x), n in np.ndenumerate(laplacian):
       if x != 0 and x < (N-1) and y != 0 and y < (M-1): 
           tmp = np.zeros((1,4))   
           tmp[0][0] = laplacian[y-1][x-1] + laplacian[y-1][x] + laplacian[y][x-1] + laplacian[y][x]
           tmp[0][1] = laplacian[y][x-1] + laplacian[y][x] + laplacian[y+1][x-1] + laplacian[y+1][x]
           tmp[0][2] = laplacian[y-1][x] + laplacian[y-1][x+1] + laplacian[y][x] + laplacian[y][x+1]
           tmp[0][3] = laplacian[y][x] + laplacian[y][x+1] + laplacian[y+1][x] + laplacian[y+1][x+1]
           maxValue = max(tmp[0])
           minValue = min(tmp[0])
           if (maxValue>0 and minValue<0):
               edge[y][x] = 255
           else :
               edge[y][x] = 0
    return edge
    
def first_prewitt(im):
    prewit_x = np.zeros(im.shape)
    filters.prewitt(im, 1, prewit_x)
    prewit_y = np.zeros(im.shape)
    filters.prewitt(im, 0, prewit_y)
    prewitt = np.sqrt(prewit_x*prewit_x + prewit_y*prewit_y) 
    return prewitt

def first_sobel(im):
    sobel_x = np.zeros(im.shape)
    filters.sobel(im, 1, sobel_x)
    sobel_y = np.zeros(im.shape)
    filters.sobel(im, 0, sobel_y)
    sobel = np.sqrt(sobel_x*sobel_x + sobel_y*sobel_y) 
    return sobel
    
def plot_im(ind, im, title):
    plt.subplot(ind)
    plt.imshow(im)
    plt.title(title)
    plt.axis('off')
    
#compare the difference of the first order and second order operator, and template
def show_difference(im):    
    laplacian0 = signal.convolve2d(im, laplacian_mask2, mode='same')
    laplacian = LoG(temp_size, sigma,im)    
    plt.gray()
    ind = 231
    plt.figure()
    plot_im(ind, im, "original")
    plot_im(ind+1, laplacian0, "laplacian_8")
    plot_im(ind+2, zero_crossing(laplacian0),"not_smoothed")
    plot_im(ind+3, zero_crossing(laplacian), "LoG edge")
    plot_im(ind+4, first_prewitt(im),"Prewitt")
    plot_im(ind+5, first_sobel(im),"Sobel")

def show_difference_l(im_l):    
    laplacian0 = signal.convolve2d(im_l, laplacian_mask2, mode='same')
    laplacian = LoG(15, 2.3,im_l)
    plt.gray()
    ind = 111
    plt.figure()
    plot_im(ind, first_prewitt(im_l),"Prewitt")
    plt.figure()
    plot_im(ind, first_sobel(im_l),"Sobel")
    plt.figure()
    plot_im(ind, zero_crossing(laplacian0),"Laplacian without smoothed")
    plt.figure()
    plot_im(ind, zero_crossing(laplacian), "LoG edge")


im_l = np.array(Image.open('lena_std.png').convert('L'))
show_difference(im)
show_difference_l(im_l)
