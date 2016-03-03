# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:55:59 2016

@author: channerduan
"""

from PIL import Image
import numpy as np
import scipy as sc
import math
import matplotlib.pyplot as plt
from timer import Timer

import cv2
from numba import double
from numba.decorators import jit, autojit

def readImage(path):
    im = np.array(Image.open(path).convert('L'))
    return im.astype(np.uint8),np.array(Image.open(path))
    
def draw(img,size=7):
    plt.figure(figsize=(size, size))
    plt.axis('off')
    plt.imshow(img)
    return
    
def createLaplacianTemplate(size,complex_=True):
    template=np.zeros((size,size),dtype=np.float)
    c = np.floor(size/2)
    if complex_:
        template[:] = -1
        template[c,c] = np.int(np.power(size,2)-1)
    else:
        template[:,c] = -1
        template[c,:] = -1
        template[c,c] = 4*c
    return template

def expandMatrix(matrix,size):
    result = np.zeros((size[0],size[1]),dtype=matrix.dtype)
    cx = np.float(matrix.shape[0]/2)
    cy = np.float(matrix.shape[1]/2)
    ccx = np.float(result.shape[0]/2)
    ccy = np.float(result.shape[1]/2)    
    result[ccx-cx:ccx+cx+1,ccy-cy:ccy+cy+1] = matrix
    return result

def imageFourierFilter(src,template):
    res = np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(src)*np.fft.fft2(expandMatrix(template,src.shape))))
    return np.real(res)

def imageFilter(src,template):
    s = src.shape
    s1 = template.shape
    c = np.int(np.floor(s1[0]/2))
    image = np.zeros((s[0],s[1]),dtype=np.int)
    for i in range(c,s[0]-c):
        for j in range(c,s[1]-c):
            image[i][j] = np.sum(src[i-c:i+c+1,j-c:j+c+1] * template)
    return image
    
def imgTransToGrayscale(src):
    min_ = np.min(src)
    max_ = np.max(src)
    res = np.round((src-min_)/(max_-min_)*255)
    return res.astype(np.uint8)    
    
def checkPos(im,posList,stroke=3,shade=200):
    for pos in posList:
        im[pos[0]-stroke:pos[0]+stroke+1,pos[1]-stroke:pos[1]+stroke+1] = shade
    return im
    
def outputImage(img,shape,isOriginal=False):
    show = cv2.resize(img,shape,interpolation=cv2.INTER_CUBIC)
    if not isOriginal:
        show = cv2.applyColorMap(imgTransToGrayscale_numba(show),cv2.COLORMAP_HOT)
    cv2.imshow('Heat FLow',show) 


scale_factor = 3

lambo1 = 1.0    # regional statistics, interior
lambo2 = 1.0    # regional statistics, exterior
k = 0.2
q = 100000000
#iter_max = 2000
iter_step = 30
free_threshold = 4
template = -createLaplacianTemplate(3,False)

imgTransToGrayscale_numba = autojit(imgTransToGrayscale)

posList = []

def output_raw(img,posList):
    color_ = (0,0,0)
    shape = (gray.shape[1]*scale_factor,gray.shape[0]*scale_factor)
    show = cv2.resize(img,shape,interpolation=cv2.INTER_CUBIC)
    for pos in posList:
        cv2.circle(show,(pos[1]*scale_factor,pos[0]*scale_factor),5,color_,-1)
    cv2.imshow('Heat FLow',show) 

# mouse callback function
def heat_injection(event,x,y,flags,param):
    global posList
    if event == cv2.EVENT_LBUTTONUP:
        x = x/scale_factor
        y = y/scale_factor
        print 'inject:%f,%f' %(y,x)
        posList.append((y,x))
        output_raw(image,posList)
def do_nothing(event,x,y,flags,param):
    return
        
plt.gray()
cv2.namedWindow('Heat FLow')

gray,image = readImage('shapes.png')
#gray,image = readImage('cherry.png')


show_img_size = (gray.shape[1]*scale_factor,gray.shape[0]*scale_factor)
#draw(checkPos(gray.copy(),posList))
draw(gray)

exit_ = False
while 1:
    posList = []
    I = np.zeros(gray.shape,np.float)
    iter_accu = 0
#    is_First = True
    output_raw(image,posList)
    cv2.setMouseCallback('Heat FLow',heat_injection)
    key = cv2.waitKey()
    c = chr(key&255)
    if c in ['q']:
        break
    if c in ['c']:
        continue
    cv2.setMouseCallback('Heat FLow',do_nothing)
    for pos in posList:
        I[pos] += q
    print 'Simulation start...'
    while 1:
        for t in range(iter_step):
            I_round = np.round(I)
            theta1 = lambo1 * (gray-np.mean(gray[(I_round>0)]))**2
            theta2 = lambo2 * (gray-np.mean(gray[(I_round<=0)]))**2
            CF = theta2 >= theta1
            delta = k*imageFourierFilter(I,template)
#            if not is_First or t > free_threshold:
            delta = delta*CF
            I += delta
#        is_First = False
        iter_accu += iter_step
        print "flow volume: %f (%d)" %(np.sum(np.abs(delta)),iter_accu)
        
        outputImage(I.copy(),show_img_size)
        key = cv2.waitKey(100)
        c = chr(key&255)
        if c in ['q']:
            exit_ = True
            break
        if c in ['c']:
            break
    if exit_:
        res = np.ones(gray.shape,np.uint8)
        res = res*(np.round(I)>0)*255
        draw(res)
        break
cv2.destroyAllWindows()



