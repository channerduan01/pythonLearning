# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 22:31:42 2016

@author: channerduan
"""

from PIL import Image
import numpy as np
import scipy as sc
import math
import matplotlib.pyplot as plt
from timer import Timer

from numba import double
from numba.decorators import jit, autojit

def readImage(path):
    '''Read image as np array, from path'''
    im = np.array(Image.open(path).convert('L'))
    return im
    
def readImages(prefix,s=0,e=3,suffix='png'):
    list_ = []
    for i in range(s,e):
        path = '%s%d.%s' %(prefix,i,suffix)
        list_.append(np.array(Image.open(path).convert('L')))
    return list_
    
def draw(img,size=7):
    plt.figure(figsize=(size, size))
    plt.axis('off')
    plt.imshow(img)
    return
    
def drawFigures(params):
    length = len(params)
    if (length < 2 or length%2 == 1):
        raise Exception("illegal input")
    for i in range(0,length,4):
        plt.figure()
        plt.subplot(121)
        plt.title(params[i])
        plt.axis('off')
        plt.imshow(params[i+1])
        if (i+2 < length):
            plt.subplot(122)
            plt.title(params[i+2])
            plt.axis('off')
            plt.imshow(params[i+3])
    return
    
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
    res = np.real(res)
    return res
    
# time_gap should be 0-0.25 for stable and slower convergence
def anisotropic(image,time_gap=0.15,iterations=20,q=10):
    img = image.copy().astype(np.int)
    deno = q**2
    s = img.shape
    tmp = np.zeros(s,img.dtype)
    for i in range(iterations):
        for x in range(1,s[0]-1):
            for y in range(1,s[1]-1):
                NI = img[x-1,y]-img[x,y]
                SI = img[x+1,y]-img[x,y]
                EI = img[x,y-1]-img[x,y]
                WI = img[x,y+1]-img[x,y]
                cN = math.exp(-NI**2/deno)
                cS = math.exp(-SI**2/deno)
                cE = math.exp(-EI**2/deno)
                cW = math.exp(-WI**2/deno)
#                tmp[x,y] = img[x,y]+time_gap*(NI+SI+EI+WI)
                tmp[x,y] = img[x,y]+time_gap*(cN*NI+cS*SI+cE*EI+cW*WI)
        img = tmp
    return img

def imgTransToGrayscale(src):
    min_ = np.min(src)
    max_ = np.max(src)
    res = np.round((src-min_)/(max_-min_)*255)
    return res.astype(np.uint8)

def sobel(image,threshold=50):
    im = image.astype(np.float)
    temp = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    #    temp = np.array([[1,2,0,-2,-1],[4,8,0,-8,-4],[6,12,0,-12,-6],[4,8,0,-8,-4],[1,2,0,-2,-1]])
    test1 = imageFourierFilter_numba(im,temp)
    test2 = imageFourierFilter_numba(im,temp.T)    
    
    mag = np.sqrt(test1**2+test2**2)
    min_ = np.min(mag)
    max_ = np.max(mag)
    mag = np.round((mag-min_)/(max_-min_)*255)
    mag = mag.astype(np.uint8)
    
    x,y = sc.where(mag <= threshold)
    mag[x,y] = 0
    test1[x,y] = 0
    test2[x,y] = 0

#    test1 = cv2.Sobel(im,cv2.CV_16S,1,0)    
#    test2 = cv2.Sobel(im,cv2.CV_16S,0,1)
#    absX = cv2.convertScaleAbs(test1)   
#    absY = cv2.convertScaleAbs(test2)  
#    mag = cv2.addWeighted(absX,0.5,absY,0.5,0)  
    
    return mag,test1,test2
    
def heatflow(data,k=0.3,max_iteration=300,stop_threshold=1):
    list_change = []
    l = len(data)
    s = data[0].shape
    HFOs = np.zeros((l-2,s[0],s[1]),np.float)
    delta = np.zeros((l-2,s[0],s[1]),np.float32)
    M,N = s
    for t in range(max_iteration):
        for i in range(1,l-1):
            delta_ = k*(data[i+1]+data[i-1]-2*data[i])
            HFOs[i-1] += np.abs(delta_*(delta_<0))
            delta[i-1] = delta_
            print 'heat change volume: %f' %np.sum(np.abs(delta))
        data[1:l-1] += delta
        change = np.sum(np.abs(delta))
        list_change.append(change)
        if t != 1 and change < stop_threshold:
            print 'exit loop at:%d' %t
            break
    res = np.zeros(HFOs.shape,np.uint8)
    for i in range(len(HFOs)):
        res[i] = imgTransToGrayscale(HFOs[i])
    return res,list_change

def get_coordinates(angle):
    delta = 0.000000000000001
    x1 = np.ceil(math.cos(angle+math.pi/8.0)*np.sqrt(2.0)-0.5-delta)
    y1 = np.ceil(-math.sin(angle-math.pi/8.0)*np.sqrt(2.0)-0.5-delta)   
    x2 = np.ceil(math.cos(angle-math.pi/8.0)*np.sqrt(2.0)-0.5-delta)    
    y2 = np.ceil(-math.sin(angle+math.pi/8.0)*np.sqrt(2.0)-0.5-delta)
    return (x1,y1,x2,y2)
    
def nms(src,x_,y_,threshold=50):
#    direct = sc.arctan(x_/y_)
    s = src.shape
    get_coordinates_numba = autojit(get_coordinates)
    for x in range(1,s[0]-1):
        for y in range(1,s[1]-1):
            if src[x,y] <= threshold:
                continue
            if y_[x,y] == 0:
                if x_[x,y] == 0:
                    angle = -math.pi/2.0
                else:
                    angle = math.pi/2.0
            else:
                angle = math.atan(x_[x,y]/y_[x,y])
            coords = get_coordinates_numba(angle)
            M1 = y_[x,y]*src[x+coords[0],y+coords[1]] + (x_[x,y]-y_[x,y])*src[x+coords[2],y+coords[3]]
            coords = get_coordinates_numba(angle+math.pi)
            M2 = y_[x,y]*src[x+coords[0],y+coords[1]] + (x_[x,y]-y_[x,y])*src[x+coords[2],y+coords[3]]
            M = src[x,y]*x_[x,y]
            if not ((M >= M1 and M >= M2) or (M <= M1 and M <= M2)):
                src[x,y] = 0
    return src

def hysterConnect(src,s,x_,y_,low,stride=1):
    for x in range(x_-stride,x_+stride+1):
        for y in range(y_-stride,y_+stride+1):
            if x >= 0 and x < s[0] and y >= 0 and y < s[1] and src[x,y] != 255 and src[x,y] >= low:
                src[x,y] = 255
                hysterConnect(src,s,x,y,low)
    return

def hysteresis(src,upper,low):
    src -= (src==255).astype(np.uint8)
    s = src.shape
    hysterConnect_numba = autojit(hysterConnect)
    for x in range(1,s[0]):
        for y in range(1,s[1]):
            if src[x,y] >= upper and src[x,y] != 255:
                src[x,y] = 255
                hysterConnect_numba(src,s,x,y,low)
#    return src
    return src*(src==255)
    
direct = None

def basic_test_demo(inputfile='ball0.png'):
    global direct
    image = readImage(inputfile)
#    image = image + np.random.normal(0,10,image.shape)
#    draw(image)
#    with Timer() as t:
#        test = anisotropic_numba(image,0.5,20,10)
#    print 'anisotropic time cost:%.2fs' % t.secs
    
    test = image    
    
#    draw(test)
    test,y_,x_ = sobel(test)
    
    
#    for (x,y),n in np.ndenumerate(direct):
#        direct[x,y] = math.atan(direct[x,y])
    draw(test)
#    draw(nms(test,direct))
#    draw(hysteresis_numba(test,120,100))
    return test


plt.gray()
anisotropic_numba = autojit(anisotropic)
imageFourierFilter_numba = autojit(imageFourierFilter)
nms_numba = autojit(nms)
hysteresis_numba = autojit(hysteresis)
heatflow_numba = autojit(heatflow)

#test = basic_test_demo()
#image = readImage('ball0.png')
#test,x_,y_ = sobel(image)
#draw(test)
#direct = sc.arctan(x_/y_)
#test = nms_numba(test.copy(),direct,x_,y_)
#draw(test)
#test = hysteresis_numba(test.copy(),100,40)
#draw(test)

#imageList = readImages('ball',0,3)
imageList = readImages('ball',0,8)
#imageList = readImages('car',0,8)

#imageList = readImages('e')

data = np.zeros((len(imageList),imageList[0].shape[0],imageList[0].shape[1]),np.float32)
grad_x = np.zeros((len(imageList),imageList[0].shape[0],imageList[0].shape[1]),np.float32)
grad_y = np.zeros((len(imageList),imageList[0].shape[0],imageList[0].shape[1]),np.float32)


num = 0
for im in imageList:
#    mag,direct = sobel(im)
    mag,x_,y_ = sobel(anisotropic_numba(im,0.1,20,2))
    data[num] = mag.astype(np.float32)
    grad_x[num] = x_
    grad_y[num] = y_   
    num += 1
HFOs,list_change = heatflow(data.copy())
for i in range(len(HFOs)):
#    draw(HFOs[i])
    x_ = grad_x[i+1]
    y_ = grad_y[i+1]
    test = nms_numba(HFOs[i].copy(),x_,y_)
#    draw(test)
#    threshold = np.mean(test) * 11
#    draw(hysteresis_numba(test.copy(),threshold,threshold/4))
    draw(hysteresis_numba(test.copy(),90,60))   
